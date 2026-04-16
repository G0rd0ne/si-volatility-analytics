"""
02_moex_client_contracts.py
Cell ID: eVcfSVEFZJlh
Exported: 2026-04-16T10:12:23.218576
"""

"""
Si Dual-Contract Volatility State v2.6.0 - Cell 2/5
HTTP Client, Contract Logic, MOEX API
"""

# ════════════════════════════════════════════════════════════════
# HTTP CLIENT
# ════════════════════════════════════════════════════════════════
class MoexClient:
    """MOEX ISS HTTP клиент с retry и connection pooling."""

    def __init__(self, cfg: MoexConfig) -> None:
        self._cfg = cfg
        self._session = requests.Session()
        self._session.headers.update({
            "User-Agent": f"si-vol-analytics/{ENGINE_VERSION} (python-requests)"
        })

    def get(self, url: str, params: Optional[dict] = None) -> requests.Response:
        """GET с экспоненциальным retry."""
        last_exc: Optional[Exception] = None
        for attempt in range(1, self._cfg.retry + 1):
            try:
                r = self._session.get(url, params=params, timeout=self._cfg.timeout)
                r.raise_for_status()
                return r
            except requests.RequestException as exc:
                last_exc = exc
                if attempt == self._cfg.retry:
                    log.error("HTTP failed after %d attempts: %s | %s", attempt, exc, url)
                    raise
                delay = 1.5 * attempt
                log.warning("HTTP retry %d/%d in %.1fs: %s", attempt, self._cfg.retry, delay, exc)
                time.sleep(delay)
        raise RuntimeError("unreachable") from last_exc

    def close(self) -> None:
        self._session.close()


# ════════════════════════════════════════════════════════════════
# TICKER / EXPIRY
# ════════════════════════════════════════════════════════════════
def make_ticker(month: int, year: int) -> str:
    return f"Si{CODE_BY_MONTH[month]}{year % 10}"

def parse_ticker(secid: str, today: date) -> tuple[int, int]:
    code, digit = secid[-2], int(secid[-1])
    month = MONTH_CODES[code]
    decade_base = (today.year // 10) * 10
    year = decade_base + digit
    if year < today.year - 2:
        year += 10
    return month, year

def _third_thursday(year: int, month: int) -> date:
    first = date(year, month, 1)
    first_thu = first + timedelta(days=(3 - first.weekday()) % 7)
    return first_thu + timedelta(weeks=2)

def fetch_lsttrade(client: MoexClient, cfg: MoexConfig, secid: str) -> Optional[date]:
    url = f"{cfg.base_url}/securities/{secid}.json"
    try:
        r = client.get(url, {"iss.meta": "off"})
        for row in r.json().get("description", {}).get("data", []):
            if len(row) >= 3 and row[0] == "LSTTRADE":
                return datetime.strptime(row[2], "%Y-%m-%d").date()
    except Exception as exc:
        log.warning("LSTTRADE failed for %s: %s", secid, exc)
    return None

def get_expiry(client: MoexClient, cfg: MoexConfig, secid: str, today: date) -> date:
    real = fetch_lsttrade(client, cfg, secid)
    if real:
        log.debug("%s LSTTRADE = %s", secid, real)
        return real
    m, y = parse_ticker(secid, today)
    est = _third_thursday(y, m)
    log.debug("%s estimated = %s", secid, est)
    return est


# ════════════════════════════════════════════════════════════════
# CONTRACT IDENTIFICATION
# ════════════════════════════════════════════════════════════════
def identify_contracts(client: MoexClient, cfg: MoexConfig, today: date) -> dict[str, ContractMeta]:
    """Идентифицирует F1, F2, F0..F-10. O(N log N)."""
    pairs = [(y, m) for y in range(today.year - 5, today.year + 3) for m in Q_MONTHS]
    tickers = [make_ticker(m, y) for y, m in pairs]
    expiries = {t: get_expiry(client, cfg, t, today) for t in tickers}
    ordered: list[tuple[str, date]] = sorted(expiries.items(), key=lambda x: x[1])

    f1_idx = next((i for i, (_, exp) in enumerate(ordered) if exp >= today), None)
    if f1_idx is None:
        raise AnalyticsError("No active Si contract found")

    result: dict[str, ContractMeta] = {}
    for offset, role in [(0, "F1"), (1, "F2")]:
        idx = f1_idx + offset
        if idx < len(ordered):
            result[role] = ContractMeta(ticker=ordered[idx][0], expiry=ordered[idx][1])
    for offset in range(1, 12):
        role = "F0" if offset == 1 else f"F-{offset-1}"
        idx = f1_idx - offset
        if idx >= 0:
            result[role] = ContractMeta(ticker=ordered[idx][0], expiry=ordered[idx][1])

    for req in ("F0", "F1", "F2"):
        if req not in result:
            raise AnalyticsError(f"Cannot identify {req} contract for {today}")

    return result

print("Cell 2/5: MOEX Client & Contracts загружены")
