#!/usr/bin/env python3
"""
gNB sensing tracer — Python port of gnb_sensing.c.

Subscribes to GNB_PHY_UL_FREQ_CHANNEL_ESTIMATE (per-UE, per-(antenna,port)
SRS frequency-domain channel estimate), buffers M consecutive captures
per (rnti, antenna, port), and computes the 2D delay-Doppler map

    h(tau, nu) = FFT_m { IFFT_k { H(k, m) } }

    k : subcarrier  (frequency -> delay   via IFFT)
    m : capture idx (slow-time -> Doppler via FFT)

GUI: two-pane matplotlib window. Top: live SRS channel magnitude for the
selected (rnti, antenna, port). Bottom: delay-Doppler heatmap, redrawn
each time a new block is ready. Keys: 'n' / 'p' cycle the displayed UE.

Usage:
    gnb_sensing.py -d <T_messages.txt> [-ip HOST] [-p PORT] [-M 32]
                                     [--save trace.gst]
    gnb_sensing.py -d <T_messages.txt> --replay trace.gst [--replay-speed 0]
"""

import argparse
import os
import socket
import struct
import sys
import threading
import time

import numpy as np
import matplotlib

matplotlib.use("TkAgg")  # safe default; user can override with MPLBACKEND
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

# --------------------------------------------------------------------------
# tunables
# --------------------------------------------------------------------------

SENS_K_MAX = 3276          # max subcarriers we accept
DEFAULT_M = 32             # slow-time captures per FFT block
DB_FLOOR, DB_CEIL = -60.0, 0.0
MAX_UES = 16
TARGET_EVENT = "GNB_PHY_UL_FREQ_CHANNEL_ESTIMATE"

# every wire event is prefixed with struct timespec (x86_64: 2x int64)
# between the outer length and the int32 event type; the outer length
# covers timespec + type + args.  (T_defs.h enables T_SEND_TIME.)
TIMESPEC_BYTES = 16
TRACE_MAGIC = b"GST1"          # gnb sensing trace v1

# --------------------------------------------------------------------------
# database (T_messages.txt)
# --------------------------------------------------------------------------

class EventFormat:
    __slots__ = ("name", "id", "arg_types", "arg_names")

    def __init__(self, name, eid):
        self.name = name
        self.id = eid
        self.arg_types = []
        self.arg_names = []


def parse_database(path):
    """Parse T_messages.txt. Returns (name_to_id, id_to_format, total_ids)."""
    name_to_id = {}
    id_to_format = {}
    total = 0
    cur = None
    with open(path, "r") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip()
            if key == "ID":
                cur = EventFormat(value, total)
                name_to_id[value] = total
                id_to_format[total] = cur
                total += 1
            elif key == "FORMAT" and cur is not None:
                # "int,gNB_ID : int,rnti : ... : buffer,chest_f"
                for tok in value.split(":"):
                    tok = tok.strip()
                    if not tok:
                        continue
                    t, _, n = tok.partition(",")
                    cur.arg_types.append(t.strip())
                    cur.arg_names.append(n.strip())
    return name_to_id, id_to_format, total


# --------------------------------------------------------------------------
# wire protocol (matches event.c / utils.c)
# --------------------------------------------------------------------------

def read_full(sock, n):
    buf = bytearray()
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            raise ConnectionError("peer closed")
        buf += chunk
    return bytes(buf)


def parse_event(body, fmt):
    """Decode the args of one event per its FORMAT. Returns dict name->value."""
    out = {}
    off = 0
    for t, n in zip(fmt.arg_types, fmt.arg_names):
        if t == "int":
            out[n] = struct.unpack_from("<i", body, off)[0]
            off += 4
        elif t == "ulong":
            out[n] = struct.unpack_from("<Q", body, off)[0]
            off += 8
        elif t == "float":
            out[n] = struct.unpack_from("<f", body, off)[0]
            off += 4
        elif t == "string":
            end = body.index(b"\x00", off)
            out[n] = body[off:end].decode("utf-8", "replace")
            off = end + 1
        elif t == "buffer":
            (sz,) = struct.unpack_from("<i", body, off)
            off += 4
            out[n] = body[off:off + sz]
            off += sz
        else:
            raise ValueError(f"unhandled arg type '{t}'")
    return out


class TClient:
    """Connects to the T tracee, asks for one event, yields decoded events."""

    def __init__(self, host, port, target_eid, total_ids):
        self.host = host
        self.port = port
        self.target_eid = target_eid
        self.total_ids = total_ids
        self.sock = None

    def connect(self):
        while True:
            try:
                print(f"connecting to {self.host}:{self.port}", flush=True)
                s = socket.create_connection((self.host, self.port), timeout=5)
                # create_connection leaves the timeout on the socket, which
                # would make every recv() raise socket.timeout after 5s of
                # radio silence — restore plain blocking mode for streaming
                s.settimeout(None)
                # handshake: byte 1, int32 N_ids, int32 is_on[N_ids]
                s.sendall(struct.pack("<B", 1))
                s.sendall(struct.pack("<i", self.total_ids))
                on_off = [0] * self.total_ids
                on_off[self.target_eid] = 1
                s.sendall(struct.pack(f"<{self.total_ids}i", *on_off))
                self.sock = s
                print("connected", flush=True)
                return
            except OSError as e:
                print(f"connect failed: {e}; retry in 1s", flush=True)
                time.sleep(1)

    def events(self):
        """Generator yielding (timespec_bytes, event_id, body_bytes).
        Reconnects on drop. timespec_bytes is the raw 16-byte time stamp
        the tracee attached, kept unparsed so --save can store it verbatim."""
        while True:
            if self.sock is None:
                self.connect()
            try:
                hdr = read_full(self.sock, 4 + TIMESPEC_BYTES + 4)
                length = struct.unpack_from("<i", hdr, 0)[0]
                etype = struct.unpack_from("<i", hdr, 4 + TIMESPEC_BYTES)[0]
                body_len = length - TIMESPEC_BYTES - 4
                if body_len < 0:
                    raise ConnectionError(
                        f"bad frame: length={length} too small (etype={etype})")
                body = read_full(self.sock, body_len)
                if etype in (-1, -2):  # config chunks, skip
                    continue
                yield hdr[4:4 + TIMESPEC_BYTES], etype, body
            except socket.timeout as e:
                print(f"socket timeout (should not happen after settimeout(None)): {e}; reconnecting",
                      flush=True)
                try:
                    self.sock.close()
                except OSError:
                    pass
                self.sock = None
            except (ConnectionError, OSError) as e:
                print(f"socket error: {e}; reconnecting", flush=True)
                try:
                    self.sock.close()
                except OSError:
                    pass
                self.sock = None


# --------------------------------------------------------------------------
# trace recording / replay
# --------------------------------------------------------------------------

class TraceWriter:
    """Appends decoded frames to a binary trace file.

    Layout (all little-endian):
        magic[4] = GST1
        per frame:
            int64 tv_sec, int64 tv_nsec   (from the tracee's T_SEND_TIME)
            int32 etype
            int32 body_len
            body[body_len]
    """

    def __init__(self, path):
        if os.path.exists(path):
            raise FileExistsError(
                f"trace file '{path}' already exists; pick another name"
                f" or delete it first")
        self.f = open(path, "wb")
        self.path = path
        self.count = 0
        self.f.write(TRACE_MAGIC)

    def write(self, timespec_bytes, etype, body):
        if len(timespec_bytes) != TIMESPEC_BYTES:
            raise ValueError(f"expected {TIMESPEC_BYTES} timespec bytes, "
                             f"got {len(timespec_bytes)}")
        self.f.write(timespec_bytes)
        self.f.write(struct.pack("<ii", etype, len(body)))
        self.f.write(body)
        self.count += 1

    def close(self):
        self.f.close()
        print(f"[sensing] trace saved: {self.path}  frames={self.count}",
              flush=True)


class TraceReplayer:
    """Streams a saved trace back at its original cadence."""

    def __init__(self, path, speed):
        self.path = path
        self.speed = speed     # 1.0 = real time, 0.0 = as fast as possible

    def events(self):
        """Yields (timespec_bytes, event_id, body_bytes) once, then stops."""
        first_ts = None
        first_wall = None
        with open(self.path, "rb") as f:
            magic = f.read(len(TRACE_MAGIC))
            if magic != TRACE_MAGIC:
                raise ValueError(
                    f"'{self.path}' is not a {TRACE_MAGIC.decode()} trace file")
            n_frames = 0
            while True:
                hdr = f.read(TIMESPEC_BYTES + 8)
                if not hdr:
                    break
                if len(hdr) != TIMESPEC_BYTES + 8:
                    raise ValueError(
                        f"truncated frame header in {self.path} after "
                        f"{n_frames} frames")
                tv_sec, tv_nsec, etype, body_len = struct.unpack("<qqii", hdr)
                body = f.read(body_len)
                if len(body) != body_len:
                    raise ValueError(
                        f"truncated frame body in {self.path} on frame "
                        f"{n_frames}")
                if body_len > 64 * 1024 * 1024:
                    raise ValueError(
                        f"implausible body_len={body_len} on frame {n_frames}")

                # pace against the recorded timestamps; negative gaps
                # (out-of-order sender clock) clamp to zero
                if first_ts is None:
                    first_ts = (tv_sec, tv_nsec)
                    first_wall = time.monotonic()
                elif self.speed > 0.0:
                    rec = (tv_sec - first_ts[0]) + (tv_nsec - first_ts[1]) * 1e-9
                    wall_elapsed = time.monotonic() - first_wall
                    wait = rec / self.speed - wall_elapsed
                    if wait > 0:
                        time.sleep(wait)
                n_frames += 1
                yield hdr[:TIMESPEC_BYTES], etype, body
        print(f"[sensing] replay finished: {self.path}  frames={n_frames}",
              flush=True)


# --------------------------------------------------------------------------
# per-UE accumulation
# --------------------------------------------------------------------------

class UeState:
    __slots__ = ("rnti", "ant", "port", "K", "M", "filled", "h_buf",
                 "last_frame", "last_slot", "srs_period_f",
                 "occ_lo", "occ_hi")

    def __init__(self, rnti, ant, port, K, M):
        self.rnti = rnti
        self.ant = ant
        self.port = port
        self.K = K                  # raw buffer width (FFT size of the carrier)
        self.M = M
        self.filled = 0
        self.h_buf = None           # allocated on first event with cropped width
        self.last_frame = -1
        self.last_slot = -1
        self.srs_period_f = 0       # frames between consecutive SRS captures (0 = unknown)
        self.occ_lo = -1            # first occupied subcarrier in the raw buffer
        self.occ_hi = -1            # one past the last occupied subcarrier

    def occ_k(self):
        """Width of the occupied subcarrier band, or 0 if not yet detected."""
        return self.occ_hi - self.occ_lo if self.occ_lo >= 0 else 0

    def label(self):
        if self.occ_lo >= 0:
            return (f"rnti={self.rnti} ant={self.ant} port={self.port}"
                    f" occ=[{self.occ_lo}..{self.occ_hi - 1}]/{self.K}")
        return f"rnti={self.rnti} ant={self.ant} port={self.port} K={self.K}"


# --------------------------------------------------------------------------
# main app
# --------------------------------------------------------------------------

class SensingApp:
    def __init__(self, M, db_fmt, scs_hz):
        self.M = M
        self.fmt = db_fmt
        self.scs_hz = scs_hz                    # subcarrier spacing for delay axis
        self.lock = threading.Lock()
        self.ues = {}            # (rnti, ant, port, K) -> UeState
        self.ue_order = []       # stable key order for prev/next cycling
        self.current_idx = 0
        self.ev_count = 0
        self.blk_count = 0
        # rate estimation: snapshot (event count, wall time) at each report
        self.rate_since_n = 0
        self.rate_since_t = time.monotonic()

        # latest block + axes for the selected UE (None until first block)
        self.latest_dd = None    # (K, M) float dB
        self.latest_chan = None  # (K,) float dB
        self.latest_axes = None  # (delay_tap_s, doppler_hz_left, doppler_hz_right)
        self.dirty = threading.Event()

        self.stop = threading.Event()

    # --------------- network side (receiver thread) ----------------------

    def lookup_ue(self, rnti, ant, port, K):
        key = (rnti, ant, port, K)
        u = self.ues.get(key)
        if u is not None:
            return u
        if len(self.ues) >= MAX_UES:
            return None
        u = UeState(rnti, ant, port, K, self.M)
        self.ues[key] = u
        self.ue_order.append(key)
        cur = self.ue_order[self.current_idx] if self.ue_order else None
        sel = " (selected)" if cur == key else ""
        print(f"[sensing] new UE slot [{len(self.ue_order)-1}]: {u.label()}{sel}",
              flush=True)
        return u

    def handle_event(self, body):
        e = parse_event(body, self.fmt)
        rnti = e["rnti"]
        frame = e["frame"]
        slot = e["subframe"]
        ant = e["antenna"]
        port = e["port"]
        buf = e["chest_f"]
        K = len(buf) // 4  # int16 I + int16 Q

        self.ev_count += 1
        if self.ev_count <= 5 or self.ev_count % 100 == 0:
            print(f"[sensing] ev={self.ev_count} rnti={rnti} frame={frame} "
                  f"slot={slot} ant={ant} port={port} bsize={len(buf)} K={K}",
                  flush=True)
        if K <= 0 or K > SENS_K_MAX:
            if self.ev_count <= 20:
                print(f"[sensing] dropping, K={K} out of range", flush=True)
            return

        with self.lock:
            u = self.lookup_ue(rnti, ant, port, K)
            if u is None:
                print(f"[sensing] no free UE slot for {rnti=} {ant=} {port=} {K=}",
                      flush=True)
                return

            # only sanity-check that time moves forward (or wraps at 1024);
            # SRS commonly fires with a period of several frames, so a jump
            # in df is the normal case and must NOT reset the fill counter —
            # otherwise no block ever completes. Doppler bin spacing then
            # reflects the actual (fixed) inter-SRS period of the config.
            if u.last_frame >= 0:
                df = frame - u.last_frame
                # frame wraps at 1024 -> treat as continuation
                if df < -512:
                    df += 1024
                elif df > 512:
                    df -= 1024
                if df < 0:
                    if u.filled > 0:
                        print(f"[sensing] time reversal rnti={rnti} frame={frame}"
                              f" slot={slot} (was {u.last_frame}/{u.last_slot})"
                              f" reset fill={u.filled}", flush=True)
                    u.filled = 0
                elif df >= 1:
                    u.srs_period_f = df
            u.last_frame = frame
            u.last_slot = slot

            # c16_t { int16 r; int16 i; } little-endian -> complex64
            iq = np.frombuffer(buf, dtype="<i2", count=K * 2).astype(np.float32)
            h_full = iq[0::2] + 1j * iq[1::2]                # (K,)

            # Detect occupied subcarrier band on first event for this UE.
            # Outside the SRS allocation the PHY leaves the buffer zero, so
            # any non-zero magnitude marks occupancy. Crop everything else
            # so the IFFT / FFT only sees the live band -- this is what the
            # user means by "calculate the delay-Doppler domain from the
            # occupied subcarriers".
            if u.occ_lo < 0:
                nz = np.flatnonzero(np.abs(h_full) > 0.0)
                if nz.size == 0:
                    # nothing transmitted (silence frame?), skip event
                    return
                u.occ_lo = int(nz[0])
                u.occ_hi = int(nz[-1]) + 1
                u.h_buf = np.zeros((u.M, u.occ_k()), dtype=np.complex64)
                print(f"[sensing] UE {u.rnti}: occupied band"
                      f" [{u.occ_lo}..{u.occ_hi - 1}] of {u.K}"
                      f" ({u.occ_k()} active subcarriers)", flush=True)

            u.h_buf[u.filled, :] = h_full[u.occ_lo:u.occ_hi]
            u.filled += 1

            cof = self.current_ue_key()
            if cof is not None and self.ues.get(cof) is u:
                # live magnitude of latest capture (already cropped)
                mag = np.abs(u.h_buf[u.filled - 1, :]) + 1e-6
                self.latest_chan = 20.0 * np.log10(mag)

            if u.filled >= u.M:
                u.filled = 0
                self.blk_count += 1
                per = (f"{u.srs_period_f}f ({u.srs_period_f * 10:.0f} ms)"
                       if u.srs_period_f > 0 else "unknown")
                print(f"[sensing] block #{self.blk_count} ready {u.label()}"
                      f"  period={per}", flush=True)
                # Simplest DFT-based delay-Doppler estimation
                # (Sanson et al., Sec. II-B.1 "DFT-Based Method"):
                #     Z = FFT_m { IFFT_k { Dr[m, k] } }
                # IFFT of size occ_K along subcarriers (axis 1) -> delay,
                # DFT of size M along slow-time captures (axis 0) -> Doppler.
                # No filtering, no clutter removal.
                # dd = np.fft.ifft(u.h_buf, axis=1)     # (M, occ_K)
                # dd = np.fft.fft(dd, axis=0)           # (M, occ_K)
                # dd = np.fft.fftshift(dd, axes=0)      # zero Doppler centered
                # block_db = 20.0 * np.log10(np.abs(dd) + 1e-12)


                Dr = u.h_buf.transpose()
                Dr = Dr / np.max(Dr)
                DFT_Dr = np.fft.fft(Dr, axis=0)
                DFT_Dr = np.fft.fftshift(DFT_Dr, axes=0)
                Z = np.fft.ifft(DFT_Dr, axis=1)

                block_db = 20.0 * np.log10(np.abs(Z) + 1e-12)


                cof = self.current_ue_key()
                if cof is not None and self.ues.get(cof) is u:
                    # transpose -> (occ_K, M): rows = delay, cols = Doppler
                    self.latest_dd = block_db.T.copy()
                    # physical axis scales
                    delay_tap_s = 1.0 / (u.occ_k() * self.scs_hz)
                    if u.srs_period_f > 0:
                        t_cap_s = u.srs_period_f * 0.010        # frames -> seconds (10 ms per frame)
                        dop_span = 1.0 / t_cap_s                  # Hz, full unambiguous width
                        dop_res = dop_span / u.M                  # Hz per Doppler bin
                        # imshow extent = outer pixel edges; bin centers must be
                        # (-M/2 : M/2-1) * dop_res so zero Doppler is exactly
                        # on the center of bin M/2
                        dop_left = -dop_span / 2.0 - dop_res / 2.0
                        dop_right = +dop_span / 2.0 - dop_res / 2.0
                        self.latest_axes = (delay_tap_s, dop_left, dop_right)
                    else:
                        self.latest_axes = (delay_tap_s, None, None)
            self.dirty.set()

    def current_ue_key(self):
        if not self.ue_order:
            return None
        self.current_idx %= len(self.ue_order)
        return self.ue_order[self.current_idx]

    # --------------- GUI side (main thread) ------------------------------

    def cycle_ue(self, step):
        with self.lock:
            if not self.ue_order:
                print("[sensing] cycle ignored: no UEs tracked yet", flush=True)
                return
            old_idx = self.current_idx
            self.current_idx = (self.current_idx + step) % len(self.ue_order)
            self.latest_dd = None
            self.latest_chan = None
            self.latest_axes = None
            old = self.ue_order[old_idx]
            new = self.ue_order[self.current_idx]
            print(f"[sensing] cycle step={step:+d}  "
                  f"{old_idx}/{len(self.ue_order)} rnti={old[0]} ant={old[1]} port={old[2]} K={old[3]}"
                  f"  ->  "
                  f"{self.current_idx}/{len(self.ue_order)} rnti={new[0]} ant={new[1]} port={new[2]} K={new[3]}",
                  flush=True)
            if len(self.ue_order) == 1:
                print("[sensing] (only 1 UE tracked; cycle is a no-op)", flush=True)
            self.dirty.set()

    def dump_ues(self, reason):
        with self.lock:
            now = time.monotonic()
            dt = now - self.rate_since_t
            dn = self.ev_count - self.rate_since_n
            self.rate_since_t = now
            self.rate_since_n = self.ev_count
            if dt > 0.0:
                rate = dn / dt
                gap_ms = dt * 1e3 / dn if dn > 0 else float("inf")
                rate_s = f"{rate:7.1f} reports/s ({gap_ms:6.1f} ms between reports)"
            else:
                rate_s = "n/a"
            cur_key = self.current_ue_key()
            print(f"[sensing] --- UE list ({reason}) --- "
                  f"tracked={len(self.ue_order)} current_idx={self.current_idx}"
                  f"  channel-report rate: {rate_s}",
                  flush=True)
            for i, k in enumerate(self.ue_order):
                u = self.ues[k]
                marker = "*" if k == cur_key else " "
                print(f"[sensing]  {marker} [{i}] rnti={u.rnti:5d} ant={u.ant}"
                      f" port={u.port} K={u.K} filled={u.filled}/{self.M}"
                      f" last={u.last_frame}/{u.last_slot}",
                      flush=True)

    def run_gui(self):
        fig, (ax_chan, ax_dd) = plt.subplots(
            2, 1, figsize=(10, 7),
            gridspec_kw={"height_ratios": [1, 2]})
        fig.canvas.manager.set_window_title("gNB sensing (python)")

        chan_line, = ax_chan.plot([], [], lw=0.7, color="#0c0c72")
        ax_chan.set_xlabel("subcarrier")
        ax_chan.set_ylabel("|H| dB")
        ax_chan.grid(True, alpha=0.3)
        chan_title = ax_chan.set_title("SRS channel — [no UE]")

        dd_norm = Normalize(DB_FLOOR, DB_CEIL)
        dd_img = ax_dd.imshow(
            np.full((16, self.M), DB_FLOOR, dtype=np.float32),
            aspect="auto", origin="lower", cmap="viridis",
            norm=dd_norm)
        ax_dd.set_xlabel("Doppler (Hz)")
        ax_dd.set_ylabel("delay (µs)")
        dd_title = ax_dd.set_title("delay-Doppler — [no UE]")
        fig.colorbar(dd_img, ax=ax_dd, label="dB")

        def on_key(ev):
            print(f"[sensing] key pressed: {ev.key!r}", flush=True)
            if ev.key in ("n", "N", "right"):
                self.cycle_ue(+1)
            elif ev.key in ("p", "P", "left"):
                self.cycle_ue(-1)
            elif ev.key in ("l", "L"):
                self.dump_ues("manual")

        fig.canvas.mpl_connect("key_press_event", on_key)

        # click on either axes also cycles, in case the backend or toolbar
        # swallows key events
        def on_click(ev):
            if ev.inaxes in (ax_chan, ax_dd):
                print(f"[sensing] mouse click in axes (button={ev.button}) — cycling",
                      flush=True)
                self.cycle_ue(+1 if ev.button != 3 else -1)

        fig.canvas.mpl_connect("button_press_event", on_click)

        def redraw(_frame):
            if not self.dirty.is_set():
                return chan_line, dd_img, chan_title, dd_title
            self.dirty.clear()
            with self.lock:
                cof = self.current_ue_key()
                lbl = "[no UE]" if cof is None else \
                    f"rnti={cof[0]} ant={cof[1]} port={cof[2]} K={cof[3]}"
                chan = None if self.latest_chan is None else self.latest_chan.copy()
                dd = None if self.latest_dd is None else self.latest_dd.copy()
                axes = self.latest_axes

            chan_title.set_text(f"SRS channel — {lbl}")
            if chan is not None:
                chan_line.set_data(np.arange(chan.size), chan)
                ax_chan.relim()
                ax_chan.autoscale_view(scalex=True, scaley=True)
            else:
                chan_line.set_data([], [])
                ax_chan.relim()
                ax_chan.autoscale_view(scalex=True, scaley=True)

            dd_title.set_text(f"delay-Doppler — {lbl}")
            if dd is not None:
                dd_img.set_data(dd)
                # autoscale the colormap to the data range with a small
                # headroom so a single strong peak doesn't blow up the scale
                lo = float(np.percentile(dd, 1.0))
                hi = float(np.percentile(dd, 99.9))
                if hi <= lo:
                    hi = lo + 1.0
                dd_norm.vmin, dd_norm.vmax = lo, hi
                n_tau = dd.shape[0]
                if axes is not None:
                    tap_s, dl, dr = axes
                    # delay axis: tap 0 at the bottom (y=0), tap n_tau-1 at the top
                    y_min_us = -0.5 * tap_s * 1e6
                    y_max_us = (n_tau - 0.5) * tap_s * 1e6
                    if dl is not None:
                        dd_img.set_extent((dl, dr, y_min_us, y_max_us))
                    else:
                        dd_img.set_extent((-dd.shape[1] / 2, dd.shape[1] / 2,
                                           y_min_us, y_max_us))
                else:
                    dd_img.set_extent((-0.5, dd.shape[1] - 0.5,
                                       -0.5, dd.shape[0] - 0.5))
            return chan_line, dd_img, chan_title, dd_title

        from matplotlib.animation import FuncAnimation
        # keep reference alive for the lifetime of the figure
        self._anim = FuncAnimation(fig, redraw, interval=100, blit=False)
        plt.tight_layout()
        plt.show()
        self.stop.set()


# --------------------------------------------------------------------------
# entry point
# --------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="gNB sensing tracer (python)")
    ap.add_argument("-d", "--database", required=True,
                    help="path to T_messages.txt")
    ap.add_argument("-ip", "--host", default="127.0.0.1")
    ap.add_argument("-p", "--port", type=int, default=2021)
    ap.add_argument("--save", metavar="FILE",
                    help="record every decoded frame to FILE so the capture "
                         "can be replayed later with --replay")
    ap.add_argument("--replay", metavar="FILE",
                    help="read frames from a previously saved trace instead of "
                         "connecting to the live socket")
    ap.add_argument("--replay-speed", type=float, default=1.0,
                    help="1.0 = follow the recorded timestamps; 0 = as fast "
                         "as possible (default 1.0)")
    ap.add_argument("-M", "--slow-time", type=int, default=DEFAULT_M,
                    help="captures per delay-Doppler block")
    ap.add_argument("--scs-hz", type=float, default=30000.0,
                    help="subcarrier spacing in Hz used to convert delay taps "
                         "to seconds (default 30 kHz; use 15000 for LTE-"
                         "like numerology, 60000/120000 for higher-µ NR)")
    ap.add_argument("--ue-dump-period", type=float, default=5.0,
                    help="seconds between periodic UE-list prints (0 disables)")
    args = ap.parse_args()

    name_to_id, id_to_format, total = parse_database(args.database)
    if TARGET_EVENT not in name_to_id:
        print(f"error: {TARGET_EVENT} not found in {args.database}",
              file=sys.stderr)
        return 1
    eid = name_to_id[TARGET_EVENT]
    fmt = id_to_format[eid]
    print(f"[sensing] target '{TARGET_EVENT}' has local id {eid}; "
          f"total ids in database: {total}", flush=True)
    need = {"rnti", "frame", "subframe", "antenna", "port", "chest_f"}
    missing = need - set(fmt.arg_names)
    if missing:
        print(f"error: {TARGET_EVENT} is missing args {sorted(missing)}",
              file=sys.stderr)
        return 1

    if args.replay and args.save:
        print("error: --save writes a fresh trace and cannot be combined "
              "with --replay", file=sys.stderr)
        return 1

    if args.replay:
        source = TraceReplayer(args.replay, args.replay_speed)
        print(f"[sensing] replaying {args.replay} at speed x{args.replay_speed}",
              flush=True)
    else:
        source = TClient(args.host, args.port, eid, total)

    writer = None
    if args.save:
        try:
            writer = TraceWriter(args.save)
        except OSError as e:
            print(f"error: cannot open trace file: {e}", file=sys.stderr)
            return 1
        print(f"[sensing] recording to {args.save}", flush=True)

    app = SensingApp(M=args.slow_time, db_fmt=fmt, scs_hz=args.scs_hz)

    seen_types = {}
    first_type_report_at = [time.time()]

    def rx_loop():
        nonlocal writer
        try:
            for timespec_bytes, etype, body in source.events():
                if app.stop.is_set():
                    return
                if writer is not None:
                    try:
                        writer.write(timespec_bytes, etype, body)
                    except OSError as e:
                        print(f"[sensing] trace write failed: {e}; "
                              f"recording disabled", flush=True)
                        writer.close()
                        writer = None
                if etype == eid:
                    try:
                        app.handle_event(body)
                    except Exception as e:
                        print(f"[sensing] handler error: {e}", flush=True)
                else:
                    seen_types[etype] = seen_types.get(etype, 0) + 1
                    if len(seen_types) == 1 and seen_types[etype] == 1:
                        print(f"[sensing] first non-target event arrived: etype={etype} "
                              f"(target {TARGET_EVENT} has id {eid}); "
                              f"body={len(body)}B", flush=True)
                    now = time.time()
                    if now - first_type_report_at[0] >= 5.0:
                        first_type_report_at[0] = now
                        if seen_types:
                            print(f"[sensing] non-target etypes seen so far: "
                                  f"{sorted(seen_types.items())}", flush=True)
        finally:
            if writer is not None:
                writer.close()
            if args.replay:
                # the trace is exhausted; keep the GUI alive so the user can
                # still inspect the last block, but expect no more updates
                print("[sensing] end of replay; GUI stays open with the "
                      "last state (close the window to exit)", flush=True)

    t = threading.Thread(target=rx_loop, daemon=True)
    t.start()

    if args.ue_dump_period > 0:
        def dump_loop():
            while not app.stop.wait(args.ue_dump_period):
                app.dump_ues("periodic")
        threading.Thread(target=dump_loop, daemon=True).start()

    app.run_gui()
    return 0


if __name__ == "__main__":
    sys.exit(main())
