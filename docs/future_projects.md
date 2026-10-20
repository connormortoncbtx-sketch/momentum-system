# Future projects (parked, not scheduled)

## Alpaca stop monitor on the droplet (parked Oct 20, 2026)
Move the Phase 2 activation check (stop -> trailing stop at +X%) off GitHub Actions polling (three dispatcher
jobs, 60-second polls) to one long-running service on the DigitalOcean droplet, driven by Alpaca's real-time
WebSocket stream instead of polling.
- Why: reliability, not speed. GitHub cron ran jobs hours late before the Netlify dispatcher, and the May 2026
  state bugs came from overlapping monitor runs. One process removes both. Speed is worth ~nothing: intraday
  stop/trail rules moved results 0.1-0.2%/trade and didn't help on holdout.
- Trigger to build it: real money going back into the stock book.
- Shape: same pattern as the Kalshi bot (install.sh, systemd service, /etc env file with Alpaca keys, ntfy alerts,
  remote stop); Connor runs the commands on his server.
