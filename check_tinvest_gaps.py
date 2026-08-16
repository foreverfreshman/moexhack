"""Проверка гэпов по данным T-Invest (как видит бот) за сегодня по всем 20 тикерам.
Запуск из D:\moex\prod:  python check_tinvest_gaps.py"""
import sys
sys.path.insert(0, '.')
from data.tinvest_stream import TInvestData, DEFAULT_UNIVERSE
from main import DEFAULT_TINKOFF_TOKEN

d = TInvestData(token=DEFAULT_TINKOFF_TOKEN, tickers=DEFAULT_UNIVERSE)
d.resolve_figis()
pc = d.get_prev_close()
so = d.get_session_open()

print("\n--- ГЭПЫ по данным T-Invest (как видит бот) ---")
gaps = []
for t in DEFAULT_UNIVERSE:
    if pc.get(t) and so.get(t):
        g = 100 * (so[t] - pc[t]) / pc[t]
        gaps.append((t, g, pc[t], so[t]))

gaps.sort(key=lambda x: -abs(x[1]))
for t, g, p, o in gaps:
    mark = "  <== ГЭП >=0.5%" if abs(g) >= 0.5 else ""
    print(f"{t:6}: prev_close={p:>9.2f}  open={o:>9.2f}  gap={g:+6.2f}%{mark}")

n_qual = sum(1 for _, g, _, _ in gaps if abs(g) >= 0.5)
print(f"\nГэпов >=0.5% (порог бота): {n_qual} из {len(gaps)}")
print(f"Если 0-1 — gap в проде почти не входит. Если 3+ — оборот будет.")
