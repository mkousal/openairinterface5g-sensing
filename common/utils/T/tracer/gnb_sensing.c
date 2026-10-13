/*
 * SPDX-License-Identifier: LicenseRef-CSSL-1.0
 *
 * gNB sensing tracer — delay-Doppler viewer.
 *
 * Subscribes to GNB_PHY_SRS_LS_ESTIMATE (per-UE, per-(antenna,port)
 * SRS least-squares channel estimate in frequency domain), buffers
 * M consecutive captures per (rnti, antenna, port), and computes
 * the 2D transform:
 *
 *   h(tau, nu) = FFT_m { IFFT_k { H(k, m) } }
 *
 *   k : subcarrier  (frequency  -> delay     via IFFT)
 *   m : capture idx (slow-time  -> Doppler   via FFT)
 *
 * No windowing — rectangular aperture. Side-lobes are expected;
 * if they hide weak paths in practice, see the Hann-window note
 * in the project memory file.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include <complex.h>
#include <pthread.h>
#include <unistd.h>
#include <fftw3.h>

#include "utils.h"
#include "event.h"
#include "handler.h"
#include "database.h"
#include "configuration.h"
#include "gui/gui.h"
#include "gui/gui_defs.h"
#include "gui/x.h"

/* buffer geometry, kept small for the sketch — tune as needed */
#define SENS_K_MAX  3276   /* max subcarriers we accept (273 RB * 12) */
#define SENS_M      32     /* slow-time captures per FFT */
#define MAX_UES     16     /* number of distinct RNTIs we track */

/* dynamic range of the heatmap, in dB */
#define DB_FLOOR    (-60.0f)
#define DB_CEIL     (0.0f)

/* ---------------------------------------------------------------- */
/* per-UE capture state                                              */
/* ---------------------------------------------------------------- */

typedef struct {
  int     used;
  int     rnti;
  int     antenna;
  int     port;
  int     last_frame;
  int     last_slot;
  int     K;            /* actual subcarrier count for this UE  */
  int     filled;       /* columns written into h_buf, 0..M-1 */
  /* h_buf stored column-major: column m of length K at h_buf[m*K] */
  fftwf_complex *h_buf; /* size K * M                         */
} sensing_ue;

/* ---------------------------------------------------------------- */
/* custom heatmap widget                                             */
/*                                                                   */
/* We own our internal struct and a paint/hints pair. We rebuild    */
/* the X pixmap only when a new delay-Doppler block is ready, then   */
/* call widget_dirty() — same pattern as the rest of the GUI.        */
/* ---------------------------------------------------------------- */

typedef struct {
  struct widget common;
  int      width;
  int      height;
  void     *ximg;             /* opaque x_image, NULL until first block */
  unsigned char *rgba;        /* width * height * 4 staging buffer     */
} heatmap_widget;

static void heatmap_paint(gui *_gui, widget *_w)
{
  struct gui *g = _gui;
  heatmap_widget *w = (heatmap_widget *)_w;
  if (w->ximg == NULL) return;   /* nothing to draw yet */
  x_draw_image(g->x, g->xwin, w->ximg, w->common.x, w->common.y);
}

static void heatmap_hints(gui *_gui, widget *_w, int *width, int *height)
{
  heatmap_widget *w = (heatmap_widget *)_w;
  *width  = w->width;
  *height = w->height;
}

static widget *new_heatmap(gui *_gui, int width, int height)
{
  struct gui *g = _gui;
  heatmap_widget *w;

  glock(g);
  w = new_widget(g, IMAGE /* we reuse the type; gui only cares about
                          * the vtable */, sizeof(heatmap_widget));
  w->width    = width;
  w->height   = height;
  w->ximg     = NULL;
  w->rgba     = malloc((size_t)width * height * 4);
  if (w->rgba == NULL) abort();
  memset(w->rgba, 0, (size_t)width * height * 4);
  w->common.paint = heatmap_paint;
  w->common.hints = heatmap_hints;
  gunlock(g);
  return (widget *)w;
}

/* viridis-ish, 256-entry cold->hot colormap approximated on the fly */
static void colormap(float t, unsigned char *r, unsigned char *g, unsigned char *b)
{
  if (t < 0) t = 0;
  if (t > 1) t = 1;
  /* simple hot-black-body ramp; replace with viridis LUT if preferred */
  *r = (unsigned char)(255 * (t < 0.33 ? t / 0.33 : 1.0));
  *g = (unsigned char)(255 * (t < 0.33 ? 0.0
                              : (t < 0.66 ? (t - 0.33) / 0.33 : 1.0)));
  *b = (unsigned char)(255 * (t < 0.66 ? 0.0 : (t - 0.66) / 0.34));
}

/* Push one delay-Doppler block (K rows x M cols, dB) into the widget. */
static void heatmap_set_block(gui *_gui, widget *_w,
                              const float *db, int K, int M,
                              float db_floor, float db_ceil)
{
  struct gui *g = _gui;
  heatmap_widget *w = (heatmap_widget *)_w;
  int W = w->width;
  int H = w->height;
  int x, y, ki, mi;

  glock(g);

  /* draw: x = Doppler (0..M-1, fftshifted so 0 is in the middle)
   *       y = delay   (0..K-1, tap 0 at top)
   */
  for (y = 0; y < H; y++) {
    ki = (int)((long)y * K / H);
    for (x = 0; x < W; x++) {
      mi  = (int)((long)x * M / W);
      float v = db[ki * M + mi];
      float t = (v - db_floor) / (db_ceil - db_floor);
      unsigned char r, gc, b;
      colormap(t, &r, &gc, &b);
      /* little-endian RGBA matches what XCreateImage expects on
       * the TrueColor visual used in x.c */
      w->rgba[4 * (y * W + x) + 0] = b;
      w->rgba[4 * (y * W + x) + 1] = gc;
      w->rgba[4 * (y * W + x) + 2] = r;
      w->rgba[4 * (y * W + x) + 3] = 255;
    }
  }

  if (w->ximg != NULL) {
    /* TODO: x_destroy ximg if/when that helper exists; leak is bounded
     * (one pixmap per FFT block) but worth fixing for long runs. */
  }
  w->ximg = x_create_image(g->x, w->rgba, W, H);
  widget_dirty(_gui, _w);
  gunlock(g);
}

/* ---------------------------------------------------------------- */
/* tracer state                                                      */
/* ---------------------------------------------------------------- */

typedef struct {
  widget *heatmap;
  widget *current_ue_label;
  widget *prev_ue_button;
  widget *next_ue_button;
  widget *status_label;
} sensing_gui;

typedef struct {
  int           socket;
  int           *is_on;
  int           nevents;
  pthread_mutex_t lock;
  void          *database;
  event_handler *h;
  gui           *g;
  sensing_gui   *ug;

  sensing_ue    ues[MAX_UES];
  int           current;          /* index into ues[] displayed   */

  /* FFTW plans, K-dependent so recreated per UE if K differs        */
  fftwf_plan    plan_k;           /* batched IFFT along k           */
  fftwf_plan    plan_m;           /* batched FFT  along m           */
  int           plan_K;           /* K the plans were built for     */
  fftwf_complex *work;            /* K * M scratch                  */
  float         *db;              /* K * M magnitude-dB output      */
} sensing_data;

/* ---------------------------------------------------------------- */
/* helpers                                                           */
/* ---------------------------------------------------------------- */

static sensing_ue *lookup_ue(sensing_data *d, int rnti, int ant, int port, int K)
{
  int i, free_i = -1;
  for (i = 0; i < MAX_UES; i++) {
    if (d->ues[i].used &&
        d->ues[i].rnti == rnti && d->ues[i].antenna == ant &&
        d->ues[i].port == port && d->ues[i].K == K)
      return &d->ues[i];
    if (!d->ues[i].used && free_i < 0) free_i = i;
  }
  if (free_i < 0) return NULL;
  sensing_ue *u = &d->ues[free_i];
  memset(u, 0, sizeof(*u));
  u->used     = 1;
  u->rnti     = rnti;
  u->antenna  = ant;
  u->port     = port;
  u->K        = K;
  u->last_frame = -1;
  u->last_slot  = -1;
  u->h_buf = fftwf_malloc(sizeof(fftwf_complex) * K * SENS_M);
  if (u->h_buf == NULL) abort();
  memset(u->h_buf, 0, sizeof(fftwf_complex) * K * SENS_M);
  return u;
}

static void build_plans(sensing_data *d, int K)
{
  if (d->plan_K == K) return;
  if (d->plan_k) fftwf_destroy_plan(d->plan_k);
  if (d->plan_m) fftwf_destroy_plan(d->plan_m);
  if (d->work)   fftwf_free(d->work);
  if (d->db)     free(d->db);

  d->work = fftwf_malloc(sizeof(fftwf_complex) * K * SENS_M);
  d->db   = malloc(sizeof(float) * K * SENS_M);
  if (!d->work || !d->db) abort();

  /* batched IFFT along k (frequency -> delay), one per capture column */
  {
    int rank = 1, n[] = { K };
    int howmany = SENS_M;
    int istride = K, idist = 1;    /* column-major: stride K between els */
    int ostride = K, odist = 1;
    d->plan_k = fftwf_plan_many_dft(rank, n, howmany,
                                    d->work, NULL, istride, idist,
                                    d->work, NULL, ostride, odist,
                                    FFTW_BACKWARD, FFTW_ESTIMATE);
  }
  /* batched FFT along m (slow-time -> Doppler), one per subcarrier row */
  {
    int rank = 1, n[] = { SENS_M };
    int howmany = K;
    int istride = K, idist = 1;    /* same layout, walking columns      */
    int ostride = K, odist = 1;
    d->plan_m = fftwf_plan_many_dft(rank, n, howmany,
                                    d->work, NULL, istride, idist,
                                    d->work, NULL, ostride, odist,
                                    FFTW_FORWARD, FFTW_ESTIMATE);
  }
  if (!d->plan_k || !d->plan_m) abort();
  d->plan_K = K;
}

static void fftshift_m(fftwf_complex *b, int K, int M)
{
  /* shift half of each row to put ν=0 in the middle */
  int half = M / 2;
  fftwf_complex *tmp = malloc(sizeof(fftwf_complex) * half);
  if (!tmp) abort();
  for (int k = 0; k < K; k++) {
    fftwf_complex *row = b + (size_t)k * M;
    memcpy(tmp, row, half * sizeof(fftwf_complex));
    memmove(row, row + half, half * sizeof(fftwf_complex));
    memcpy(row + half, tmp, half * sizeof(fftwf_complex));
  }
  free(tmp);
}

/* ---------------------------------------------------------------- */
/* called once a UE buffer is full                                   */
/* ---------------------------------------------------------------- */

static void process_block(sensing_data *d, sensing_ue *u)
{
  int K = u->K;
  build_plans(d, K);

  memcpy(d->work, u->h_buf, sizeof(fftwf_complex) * K * SENS_M);

  fftwf_execute(d->plan_k);   /* delay */
  fftwf_execute(d->plan_m);   /* Doppler */

  fftshift_m(d->work, K, SENS_M);

  /* magnitude in dB */
  const float eps = 1e-12f;
  for (int i = 0; i < K * SENS_M; i++) {
    float re = crealf(d->work[i]);
    float im = cimagf(d->work[i]);
    d->db[i] = 10.f * log10f(re * re + im * im + eps);
  }

  /* only paint if this UE is the selected one — caller holds d->lock */
  if (&d->ues[d->current] == u) {
    heatmap_set_block(d->g, d->ug->heatmap, d->db, K, SENS_M,
                      DB_FLOOR, DB_CEIL);
  }
}

/* ---------------------------------------------------------------- */
/* event handler                                                     */
/* ---------------------------------------------------------------- */

/* event layout for GNB_PHY_SRS_LS_ESTIMATE:
 *   int,gNB_ID : int,rnti : int,frame : int,slot : int,antenna :
 *   int,port : buffer,chest_ls
 *
 * We resolve argument positions from the database format rather
 * than hard-coding indexes.
 */
typedef struct {
  int rnti_arg, frame_arg, slot_arg, ant_arg, port_arg, buf_arg;
} arg_pos;

static void resolve_arg_positions(void *database, char *event_name, arg_pos *p)
{
  int eid = event_id_from_name(database, event_name);
  database_event_format f = get_format(database, eid);
  memset(p, -1, sizeof(*p));
  for (int i = 0; i < f.count; i++) {
    if (!strcmp(f.name[i], "rnti"))     p->rnti_arg  = i;
    if (!strcmp(f.name[i], "frame"))    p->frame_arg = i;
    if (!strcmp(f.name[i], "slot"))     p->slot_arg  = i;
    if (!strcmp(f.name[i], "antenna"))  p->ant_arg   = i;
    if (!strcmp(f.name[i], "port"))     p->port_arg  = i;
    if (!strcmp(f.name[i], "chest_ls")) p->buf_arg   = i;
  }
  if (p->rnti_arg < 0 || p->frame_arg < 0 || p->slot_arg < 0 ||
      p->ant_arg  < 0 || p->port_arg  < 0 || p->buf_arg  < 0) {
    fprintf(stderr, "fatal: %s doesn't carry all expected args\n", event_name);
    exit(1);
  }
}

/* C has no closures; stash arg positions and tracer state as file
 * globals so the handler can reach them. */
static arg_pos g_pos;
static sensing_data *g_data;

static void srs_handler(void *unused, event e)
{
  sensing_data *d = g_data;
  (void)unused;

  int rnti = e.e[g_pos.rnti_arg].i;
  int frame = e.e[g_pos.frame_arg].i;
  int slot  = e.e[g_pos.slot_arg].i;
  int ant   = e.e[g_pos.ant_arg].i;
  int port  = e.e[g_pos.port_arg].i;

  event_arg buf = e.e[g_pos.buf_arg];
  int K = buf.bsize / (int)sizeof(int32_t);    /* chest_ls is I/Q int32   */
  if (K <= 0 || K > SENS_K_MAX) return;

  sensing_ue *u = lookup_ue(d, rnti, ant, port, K);
  if (u == NULL) return;

  /* drop data if a slot was skipped (uneven slow-time sampling) */
  if (u->last_frame >= 0) {
    int df = frame - u->last_frame;
    int ds = slot  - u->last_slot;
    if (df < 0 || df > 1 || (df == 0 && ds <= 0) || (df == 0 && ds > 20)) {
      u->filled = 0;
    }
  }
  u->last_frame = frame;
  u->last_slot  = slot;

  /* convert I/Q int32 -> float complex and store as column u->filled */
  int32_t *src = (int32_t *)buf.b;
  for (int k = 0; k < K; k++) {
    int16_t re = (int16_t)(src[k] & 0xffff);
    int16_t im = (int16_t)(src[k] >> 16);
    u->h_buf[(size_t)u->filled * K + k] = (float)re + I * (float)im;
  }
  u->filled++;

  if (u->filled >= SENS_M) {
    u->filled = 0;
    process_block(d, u);
  }
}

/* ---------------------------------------------------------------- */
/* GUI                                                               */
/* ---------------------------------------------------------------- */

static void set_current_label(sensing_data *d)
{
  char s[64];
  if (d->current < 0 || d->current >= MAX_UES || !d->ues[d->current].used) {
    label_set_text(d->g, d->ug->current_ue_label, "[no UE]");
  } else {
    snprintf(s, sizeof(s), "[UE %d ant %d port %d]",
             d->ues[d->current].rnti,
             d->ues[d->current].antenna,
             d->ues[d->current].port);
    label_set_text(d->g, d->ug->current_ue_label, s);
  }
}

static void click(void *priv, gui *g, char *notif, widget *w, void *ndata)
{
  int *dd = ndata;
  int button = dd[0];
  sensing_data *d = priv;
  if (button != 1) return;

  if (pthread_mutex_lock(&d->lock)) abort();
  int start = d->current;
  int step = 0;
  if (w == d->ug->prev_ue_button) step = -1;
  if (w == d->ug->next_ue_button) step = +1;
  if (step != 0) {
    /* walk to next used UE entry */
    for (int i = 0; i < MAX_UES; i++) {
      d->current = (d->current + step + MAX_UES) % MAX_UES;
      if (d->ues[d->current].used) break;
    }
    if (!d->ues[d->current].used) d->current = start;
    set_current_label(d);
  }
  if (pthread_mutex_unlock(&d->lock)) abort();
}

static void build_gui(sensing_data *d)
{
  gui *g = d->g;
  sensing_gui *ug = d->ug;

  widget *win = new_toplevel_window(g, 900, 540, "gNB sensing");
  widget *top = new_container(g, VERTICAL);
  widget_add_child(g, win, top, -1);

  widget *row = new_container(g, HORIZONTAL);
  widget_add_child(g, top, row, -1);

  widget *lbl = new_label(g, "[no UE]");
  label_set_clickable(g, lbl, 1);
  ug->current_ue_label = lbl;
  widget_add_child(g, row, lbl, -1);

  widget *prev = new_label(g, "  [prev UE]  ");
  label_set_clickable(g, prev, 1);
  ug->prev_ue_button = prev;
  widget_add_child(g, row, prev, -1);

  widget *next = new_label(g, "  [next UE]  ");
  label_set_clickable(g, next, 1);
  ug->next_ue_button = next;
  widget_add_child(g, row, next, -1);

  /* heatmap: x=Doppler, y=delay */
  ug->heatmap = new_heatmap(g, 880, 480);
  widget_add_child(g, top, ug->heatmap, -1);

  register_notifier(g, "click", prev, click, d);
  register_notifier(g, "click", next, click, d);
}

static void *gui_thread(void *_g)
{
  gui *g = _g;
  gui_loop(g);
  return NULL;
}

/* ---------------------------------------------------------------- */
/* main                                                              */
/* ---------------------------------------------------------------- */

void usage(void)
{
  printf(
"options:\n"
"    -d   <database file>      this option is mandatory\n"
"    -ip <host>                connect to given IP address (default %s)\n"
"    -p  <port>                connect to given port (default %d)\n",
    DEFAULT_REMOTE_IP, DEFAULT_REMOTE_PORT);
  exit(1);
}

int main(int n, char **v)
{
  char *database_filename = NULL;
  void *database;
  char *ip   = DEFAULT_REMOTE_IP;
  int   port = DEFAULT_REMOTE_PORT;
  int *is_on;
  int  number_of_events;
  event_handler *h;
  sensing_data  sd;
  sensing_gui   sg;

  for (int i = 1; i < n; i++) {
    if (!strcmp(v[i], "-h") || !strcmp(v[i], "--help")) usage();
    if (!strcmp(v[i], "-d"))  { if (i > n-2) usage(); database_filename = v[++i]; continue; }
    if (!strcmp(v[i], "-ip")) { if (i > n-2) usage(); ip = v[++i];   continue; }
    if (!strcmp(v[i], "-p"))  { if (i > n-2) usage(); port = atoi(v[++i]); continue; }
    usage();
  }
  if (database_filename == NULL) {
    printf("ERROR: provide a database file (-d)\n");
    exit(1);
  }

  database = parse_database(database_filename);
  load_config_file(database_filename);

  number_of_events = number_of_ids(database);
  is_on = calloc(number_of_events, sizeof(int));
  if (!is_on) abort();

  on_off(database, "GNB_PHY_SRS_LS_ESTIMATE", is_on, 1);

  resolve_arg_positions(database, "GNB_PHY_SRS_LS_ESTIMATE", &g_pos);

  memset(&sd, 0, sizeof(sd));
  memset(&sg, 0, sizeof(sg));
  sd.socket = -1;
  sd.is_on  = is_on;
  sd.nevents = number_of_events;
  sd.database = database;
  sd.current = 0;
  sd.ug = &sg;
  if (pthread_mutex_init(&sd.lock, NULL)) abort();

  h = new_handler(database);
  sd.h = h;

  g_data = &sd;
  register_handler_function(h, event_id_from_name(database, "GNB_PHY_SRS_LS_ESTIMATE"),
                            srs_handler, NULL);

  sd.g = gui_init();
  new_thread(gui_thread, sd.g);
  build_gui(&sd);
  set_current_label(&sd);

  OBUF ebuf = {.osize = 0, .omaxsize = 0, .obuf = NULL};

restart:
  clear_remote_config();
  if (sd.socket != -1) close(sd.socket);
  sd.socket = connect_to(ip, port);

  int t = 1;
  if (socket_send(sd.socket, &t, 1) == -1 ||
      socket_send(sd.socket, &number_of_events, sizeof(int)) == -1 ||
      socket_send(sd.socket, is_on, number_of_events * sizeof(int)) == -1)
    goto restart;

  while (1) {
    event e = get_event(sd.socket, &ebuf, database);
    if (e.type == -1) goto restart;
    if (pthread_mutex_lock(&sd.lock)) abort();
    handle_event(h, e);
    if (pthread_mutex_unlock(&sd.lock)) abort();
  }
  return 0;
}
