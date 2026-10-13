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
"""

import argparse
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
        """Generator yielding (event_id, body_bytes). Reconnects on drop."""
        while True:
            if self.sock is None:
                self.connect()
            try:
                # header: int32 length (covers the rest), int32 type
                hdr = read_full(self.sock, 8)
                length, etype = struct.unpack("<ii", hdr)
                body = read_full(self.sock, length - 4)
                if etype in (-1, -2):  # config chunks, skip
                    continue
                yield etype, body
            except (ConnectionError, OSError) as e:
                print(f"socket error: {e}; reconnecting", flush=True)
                try:
                    self.sock.close()
                except OSError:
                    pass
                self.sock = None


# --------------------------------------------------------------------------
# per-UE accumulation
# --------------------------------------------------------------------------

class UeState:
    __slots__ = ("rnti", "ant", "port", "K", "filled", "h_buf",
                 "last_frame", "last_slot")

    def __init__(self, rnti, ant, port, K, M):
        self.rnti = rnti
        self.ant = ant
        self.port = port
        self.K = K
        self.filled = 0
        self.h_buf = np.zeros((M, K), dtype=np.complex64)
        self.last_frame = -1
        self.last_slot = -1

    def label(self):
        return f"rnti={self.rnti} ant={self.ant} port={self.port} K={self.K}"


# --------------------------------------------------------------------------
# main app
# --------------------------------------------------------------------------

class SensingApp:
    def __init__(self, M, db_fmt):
        self.M = M
        self.fmt = db_fmt
        self.lock = threading.Lock()
        self.ues = {}            # (rnti, ant, port, K) -> UeState
        self.ue_order = []       # stable key order for prev/next cycling
        self.current_idx = 0
        self.ev_count = 0
        self.blk_count = 0

        # latest block + latest channel snapshot for the selected UE
        self.latest_dd = None    # (K, M) float dB
        self.latest_chan = None  # (K,) float dB
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
        print(f"[sensing] new UE slot: {u.label()}", flush=True)
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

            # drop data on a slot gap (uneven slow-time sampling)
            # note: same (frame, slot) is OK — multiple SRS symbols per slot
            # or repeat firings are still distinct captures
            if u.last_frame >= 0:
                df = frame - u.last_frame
                ds = slot - u.last_slot
                if df < 0 or df > 1 or (df == 0 and ds < 0) or (df == 0 and ds > 20):
                    if u.filled > 0:
                        print(f"[sensing] gap rnti={rnti} frame={frame} slot={slot}"
                              f" (was {u.last_frame}/{u.last_slot}) reset fill={u.filled}",
                              flush=True)
                    u.filled = 0
            u.last_frame = frame
            u.last_slot = slot

            # c16_t { int16 r; int16 i; } little-endian -> complex64
            iq = np.frombuffer(buf, dtype="<i2", count=K * 2).astype(np.float32)
            u.h_buf[u.filled, :] = iq[0::2] + 1j * iq[1::2]
            u.filled += 1

            cof = self.current_ue_key()
            if cof is not None and self.ues.get(cof) is u:
                # live magnitude of latest capture
                mag = np.abs(u.h_buf[u.filled - 1, :]) + 1e-6
                self.latest_chan = 20.0 * np.log10(mag)

            if u.filled >= self.M:
                u.filled = 0
                self.blk_count += 1
                print(f"[sensing] block #{self.blk_count} ready {u.label()}",
                      flush=True)
                # delay-Doppler: IFFT along k then FFT along m, fftshift on m
                delay = np.fft.ifft(u.h_buf, axis=1)         # (M, K)
                dd = np.fft.fft(delay, axis=0)               # (M, K)
                dd = np.fft.fftshift(dd, axes=0)
                mag = np.abs(dd).astype(np.float32) ** 2
                block_db = 10.0 * np.log10(mag + 1e-12)
                cof = self.current_ue_key()
                if cof is not None and self.ues.get(cof) is u:
                    # transpose -> (K, M): rows = delay, cols = Doppler
                    self.latest_dd = block_db.T.copy()
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
                return
            self.current_idx = (self.current_idx + step) % len(self.ue_order)
            self.latest_dd = None
            self.latest_chan = None
            print(f"[sensing] selected {self.ue_order[self.current_idx]}",
                  flush=True)
            self.dirty.set()

    def run_gui(self):
        fig, (ax_chan, ax_dd) = plt.subplots(
            2, 1, figsize=(10, 7),
            gridspec_kw={"height_ratios": [1, 2]})
        fig.canvas.manager.set_window_title("gNB sensing (python)")

        chan_line, = ax_chan.plot([], [], lw=0.7, color="#0c0c72")
        ax_chan.set_xlabel("subcarrier")
        ax_chan.set_ylabel("|H| dB")
        ax_chan.set_ylim(-20, 100)
        ax_chan.grid(True, alpha=0.3)
        chan_title = ax_chan.set_title("SRS channel — [no UE]")

        dd_img = ax_dd.imshow(
            np.full((16, self.M), DB_FLOOR, dtype=np.float32),
            aspect="auto", origin="lower", cmap="viridis",
            norm=Normalize(DB_FLOOR, DB_CEIL))
        ax_dd.set_xlabel("Doppler bin")
        ax_dd.set_ylabel("delay tap")
        dd_title = ax_dd.set_title("delay-Doppler — [no UE]")
        fig.colorbar(dd_img, ax=ax_dd, label="dB")

        def on_key(ev):
            if ev.key in ("n", "N", "right"):
                self.cycle_ue(+1)
            elif ev.key in ("p", "P", "left"):
                self.cycle_ue(-1)

        fig.canvas.mpl_connect("key_press_event", on_key)

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

            chan_title.set_text(f"SRS channel — {lbl}")
            if chan is not None:
                chan_line.set_data(np.arange(chan.size), chan)
                ax_chan.set_xlim(0, chan.size - 1)
            else:
                chan_line.set_data([], [])

            dd_title.set_text(f"delay-Doppler — {lbl}")
            if dd is not None:
                dd_img.set_data(dd)
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
    ap.add_argument("-M", "--slow-time", type=int, default=DEFAULT_M,
                    help="captures per delay-Doppler block")
    args = ap.parse_args()

    name_to_id, id_to_format, total = parse_database(args.database)
    if TARGET_EVENT not in name_to_id:
        print(f"error: {TARGET_EVENT} not found in {args.database}",
              file=sys.stderr)
        return 1
    eid = name_to_id[TARGET_EVENT]
    fmt = id_to_format[eid]
    need = {"rnti", "frame", "subframe", "antenna", "port", "chest_f"}
    missing = need - set(fmt.arg_names)
    if missing:
        print(f"error: {TARGET_EVENT} is missing args {sorted(missing)}",
              file=sys.stderr)
        return 1

    app = SensingApp(M=args.slow_time, db_fmt=fmt)
    client = TClient(args.host, args.port, eid, total)

    def rx_loop():
        for etype, body in client.events():
            if app.stop.is_set():
                return
            if etype == eid:
                try:
                    app.handle_event(body)
                except Exception as e:
                    print(f"[sensing] handler error: {e}", flush=True)

    t = threading.Thread(target=rx_loop, daemon=True)
    t.start()
    app.run_gui()
    return 0


if __name__ == "__main__":
    sys.exit(main())
