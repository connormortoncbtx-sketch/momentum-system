#!/usr/bin/env bash
# Kalshi market-making bot: one-time install on a fresh Ubuntu 24.04 server (DigitalOcean droplet).
#   curl -fsSL https://raw.githubusercontent.com/connormortoncbtx-sketch/momentum-system/main/kalshi_mm/deploy/install.sh | sudo bash
# Re-running is safe: it updates the code and restarts the service, keeping keys and settings.
set -euo pipefail
REPO=https://github.com/connormortoncbtx-sketch/momentum-system.git
APP=/home/kmm/app; DATA=/home/kmm/data; ETC=/etc/kmm

echo "==> packages"
# only touch apt when something is missing: apt waits on Ubuntu's background updates and is the slowest step
if ! dpkg -s python3-venv git curl >/dev/null 2>&1; then apt-get update -qq && apt-get install -y -qq python3-venv git curl >/dev/null; fi
# swap: a 1 GB droplet with no swap stalls (SSH timeouts, bot frozen) when memory fills; with swap it just slows down
if ! swapon --show | grep -q .; then
  echo "==> adding a 2 GB swap file"
  fallocate -l 2G /swapfile && chmod 600 /swapfile && mkswap -q /swapfile && swapon /swapfile
  grep -q '^/swapfile' /etc/fstab || echo '/swapfile none swap sw 0 0' >> /etc/fstab
fi
id kmm >/dev/null 2>&1 || useradd -m -s /bin/bash kmm
timedatectl set-timezone America/Chicago || true

echo "==> code"
OLD_REV=""
if [ -d "$APP/.git" ]; then OLD_REV=$(sudo -u kmm git -C "$APP" rev-parse HEAD 2>/dev/null || true); sudo -u kmm git -C "$APP" pull -q; else sudo -u kmm git clone -q --depth 1 "$REPO" "$APP"; fi
[ -x /home/kmm/venv/bin/python ] || sudo -u kmm python3 -m venv /home/kmm/venv
# packages only when missing (re-downloading them on every update was slow on this small server)
sudo -u kmm /home/kmm/venv/bin/python -c "import cryptography, websocket" 2>/dev/null || \
  sudo -u kmm nice -n 10 /home/kmm/venv/bin/pip install -q cryptography websocket-client
sudo -u kmm nice -n 10 /home/kmm/venv/bin/python "$APP/kalshi_mm/test_bot.py"
sudo -u kmm env LOG_DIR="$(sudo -u kmm mktemp -d)" KILL_FILE=/tmp/kmm-lottery-test-kill \
  nice -n 10 /home/kmm/venv/bin/python "$APP/kalshi_mm/test_lottery.py" | grep -v INFO
sudo -u kmm mkdir -p "$DATA"
sudo -u kmm env LOG_DIR="$DATA" nice -n 10 /home/kmm/venv/bin/python "$APP/kalshi_mm/compact_logs.py" || true

echo "==> settings ($ETC, readable by the bot user only)"
mkdir -p "$ETC"
if [ ! -f "$ETC/env" ]; then
cat > "$ETC/env" <<'EOF'
# --- Kalshi market-making bot settings. Edit with:  sudo nano /etc/kmm/env  then:  sudo systemctl restart kmm
# MODE: demo (fake money) or prod (real money). prod also needs I_UNDERSTAND_REAL_MONEY=yes
MODE=demo
I_UNDERSTAND_REAL_MONEY=no
KALSHI_KEY_ID=paste-your-key-id-here
KALSHI_PRIVATE_KEY_FILE=/etc/kmm/kalshi.pem
# risk limits: budget = BUDGET_PCT x account value (grows/shrinks with the account), capped at BUDGET_MAX_USD;
# daily loss stop = DAILY_LOSS_PCT x the day's starting value. (BUDGET_USD / DAILY_LOSS_USD apply only if the PCTs are 0.)
BUDGET_PCT=0.9
BUDGET_MAX_USD=3000
DAILY_LOSS_PCT=0.15
BUDGET_USD=200
DAILY_LOSS_USD=25
SIZE=5
INV_CAP=25
# max contracts across all buckets of one city-day (they're one temperature = one bet)
EVENT_CAP=30
OFFSET_C=2
# only add inventory on the side priced >= FAV_MIN (long-shot fills lose; 0 = off)
FAV_MIN=0.5
# seconds between cycles; QUIET_MINUTES=45-59 would quote only position-reducing sides in those minutes (off when blank)
LOOP_SECONDS=5
QUIET_MINUTES=
# USE_WS=yes: live order books pushed by Kalshi (re-quote within ~1s of a change); off = request every book each cycle
USE_WS=no
# quote only 6-48h before close (the per-fill study found the last 6h earns ~nothing)
MIN_HOURS=6
MAX_HOURS=48
# phone alerts (same ntfy channel the momentum system uses; leave blank to disable)
NTFY_CHANNEL=
LOG_DIR=/home/kmm/data
KILL_FILE=/home/kmm/data/KILL
EOF
fi
[ -f "$ETC/kalshi.pem" ] || : > "$ETC/kalshi.pem"
chown -R kmm:kmm "$ETC"; chmod 700 "$ETC"; chmod 600 "$ETC"/*

echo "==> services"
cat > /etc/systemd/system/kmm.service <<EOF
[Unit]
Description=Kalshi market-making bot
After=network-online.target
Wants=network-online.target
[Service]
User=kmm
WorkingDirectory=$DATA
EnvironmentFile=$ETC/env
ExecStart=/home/kmm/venv/bin/python $APP/kalshi_mm/bot.py
Restart=always
RestartSec=30
# the trading bot comes first: last to be killed if memory runs out, first in line for CPU
OOMScoreAdjust=-800
CPUWeight=1000
# watchdog: the bot pings systemd every loop; if it hangs for 5 min, systemd restarts it and the fresh process takes
# over the resting orders (re-prices or cancels them on its first cycle), so stale quotes can't sit for long
WatchdogSec=300
NotifyAccess=main
# exit code 3 = kill switch: stay stopped
RestartPreventExitStatus=3
[Install]
WantedBy=multi-user.target
EOF
cat > /etc/systemd/system/kmm-report.service <<EOF
[Unit]
Description=Kalshi bot daily report
[Service]
Type=oneshot
User=kmm
EnvironmentFile=$ETC/env
ExecStart=/home/kmm/venv/bin/python $APP/kalshi_mm/report.py
EOF
cat > /etc/systemd/system/kmm-dash.service <<EOF
[Unit]
Description=Kalshi bot read-only dashboard (127.0.0.1:8787; reach it via 'tailscale serve')
After=network-online.target
[Service]
User=kmm
WorkingDirectory=$DATA
EnvironmentFile=$ETC/env
ExecStart=/home/kmm/venv/bin/python $APP/kalshi_mm/dashboard.py
Restart=always
RestartSec=10
# secondary: capped memory, low CPU priority, first to go if memory runs out
MemoryMax=400M
CPUWeight=20
Nice=10
OOMScoreAdjust=500
[Install]
WantedBy=multi-user.target
EOF
cat > /etc/systemd/system/kmm-sniper.service <<EOF
[Unit]
Description=Kalshi decided-bucket PAPER shadow (reads NWS + order books, places no orders)
After=network-online.target
[Service]
User=kmm
WorkingDirectory=$DATA
EnvironmentFile=$ETC/env
ExecStart=/home/kmm/venv/bin/python $APP/kalshi_mm/sniper.py
Restart=always
RestartSec=60
MemoryMax=200M
CPUWeight=20
Nice=10
OOMScoreAdjust=500
[Install]
WantedBy=multi-user.target
EOF
cat > /etc/systemd/system/kmm-lottery.service <<EOF
[Unit]
Description=Kalshi weekly roll (lottery ticket)
[Service]
Type=oneshot
User=kmm
WorkingDirectory=$DATA
EnvironmentFile=$ETC/env
ExecStart=/home/kmm/venv/bin/python $APP/kalshi_mm/lottery.py
MemoryMax=200M
CPUWeight=50
Nice=5
EOF
cat > /etc/systemd/system/kmm-lottery.timer <<EOF
[Unit]
Description=Kalshi weekly roll every 15 minutes
[Timer]
OnCalendar=*:0/15
Persistent=true
[Install]
WantedBy=timers.target
EOF
cat > /etc/systemd/system/kmm-report.timer <<EOF
[Unit]
Description=Kalshi bot daily report (9pm Central)
[Timer]
OnCalendar=*-*-* 21:00:00
Persistent=true
[Install]
WantedBy=timers.target
EOF
cat > /usr/local/bin/kmm <<'KMM'
#!/usr/bin/env bash
# kmm status | stop | resume | logs [n] | report | positions | diagnose | lottery | update      (run with sudo)
case "${1:-status}" in
  status) systemctl --no-pager status kmm | head -5; tail -n 1 /home/kmm/data/snapshots.jsonl 2>/dev/null; echo ;;
  stop)   sudo -u kmm touch /home/kmm/data/KILL; echo "kill switch set: the bot cancels all orders within ~30s and stays stopped" ;;
  resume) rm -f /home/kmm/data/KILL; systemctl restart kmm; echo "resumed" ;;
  logs)   journalctl -u kmm -n "${2:-40}" --no-pager ;;
  report) sudo -u kmm bash -c 'set -a; . /etc/kmm/env; /home/kmm/venv/bin/python /home/kmm/app/kalshi_mm/report.py' ;;
  diagnose) sudo -u kmm bash -c 'set -a; . /etc/kmm/env; /home/kmm/venv/bin/python /home/kmm/app/kalshi_mm/diagnose.py "$@"' _ ${2:+"$2"} ;;
  positions) sudo -u kmm bash -c 'set -a; . /etc/kmm/env; /home/kmm/venv/bin/python /home/kmm/app/kalshi_mm/positions.py' ;;
  sniper) sudo -u kmm bash -c 'set -a; . /etc/kmm/env; /home/kmm/venv/bin/python /home/kmm/app/kalshi_mm/sniper.py report'; journalctl -u kmm-sniper -n 3 --no-pager | grep -v systemd ;;
  lottery) cat /home/kmm/data/lottery.json 2>/dev/null || echo "no weekly roll yet"; echo; journalctl -u kmm-lottery -n "${2:-10}" --no-pager | grep -v systemd ;;
  update) exec bash -c "curl -fsSL https://raw.githubusercontent.com/connormortoncbtx-sketch/momentum-system/main/kalshi_mm/deploy/install.sh | bash" ;;  # exec: this script gets rewritten mid-update
  *) echo "usage: kmm status|stop|resume|logs [n]|report|positions|diagnose|lottery|sniper|update" ;;
esac
KMM
chmod 755 /usr/local/bin/kmm
systemctl daemon-reload
systemctl enable -q kmm-report.timer && systemctl start kmm-report.timer
systemctl enable -q kmm-lottery.timer && systemctl start kmm-lottery.timer
systemctl enable -q kmm-dash && systemctl restart kmm-dash
systemctl enable -q kmm-sniper && systemctl restart kmm-sniper
systemctl enable -q kmm

if grep -q "paste-your-key-id-here" "$ETC/env" || [ ! -s "$ETC/kalshi.pem" ]; then
  echo
  echo "Installed. Two things left before it starts:"
  echo "  1. sudo nano /etc/kmm/env         -> set KALSHI_KEY_ID (and NTFY_CHANNEL if you want alerts)"
  echo "  2. sudo nano /etc/kmm/kalshi.pem  -> paste the private key, including the BEGIN/END lines"
  echo "Then:  sudo systemctl restart kmm   and check with:  sudo kmm logs"
elif [ -n "$OLD_REV" ] && systemctl is-active -q kmm && \
     sudo -u kmm git -C "$APP" diff --quiet "$OLD_REV" HEAD -- kalshi_mm/bot.py kalshi_mm/ws.py kalshi_mm/lotfiles.py; then
  # the trading code didn't change: leave the bot running (a restart cancels and re-places every order)
  echo "Installed. Trading bot code unchanged -- bot NOT restarted (orders untouched)."
else
  systemctl restart kmm
  echo "Installed and (re)started the trading bot."
fi
echo "Commands (with sudo): kmm status | kmm logs | kmm report | kmm positions | kmm lottery | kmm stop | kmm resume | kmm update"
echo "Dashboard: http://127.0.0.1:8787 on the server; to open it from your phone, once: install Tailscale here"
echo "  (curl -fsSL https://tailscale.com/install.sh | sh && tailscale up) and run: tailscale serve --bg 8787"
echo "Remote stop from your phone: in the ntfy app, publish 'stop' to topic <your NTFY_CHANNEL>-kmm-stop"
