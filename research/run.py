"""
Запуск полного pipeline на реальных данных MOEX.

Использование:
    python run.py --data-dir /path/to/MOEX/market_data_final
    python run.py --data-dir ... --cache cache.parquet           # с кэшем
    python run.py --data-dir ... --diagnose                       # + диагностика данных

Опционально можно подавить веса в качестве sanity-check (ставит data в parquet
для быстрых re-runs).
"""

import argparse
import logging
import sys
from pathlib import Path

# Сделаем модули видимыми
sys.path.insert(0, str(Path(__file__).parent))

from data_loader import load_moex_daily, diagnose_daily, TICKERS_DEFAULT
from pipeline import run_pipeline, HORIZON_DAYS


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True,
                        help="Папка с {TICKER}_3y_1m.csv файлами")
    parser.add_argument("--cache", default=None,
                        help="Parquet кэш дневных свечей (опционально)")
    parser.add_argument("--diagnose", action="store_true",
                        help="Печатать диагностику данных")
    parser.add_argument("--tickers", default=None,
                        help="Через запятую, например LKOH,SBER (default: все 20)")
    parser.add_argument("--file-pattern", default="{ticker}_3y_1m.csv",
                        help="Шаблон имени файла")
    parser.add_argument("--rebuild-cache", action="store_true",
                        help="Игнорировать существующий кэш")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    )
    log = logging.getLogger("run")

    tickers = args.tickers.split(",") if args.tickers else TICKERS_DEFAULT

    log.info("==== STEP 0: Load and aggregate data ====")
    daily = load_moex_daily(
        args.data_dir, tickers=tickers,
        file_pattern=args.file_pattern, cache_path=args.cache,
        rebuild_cache=args.rebuild_cache,
    )

    if args.diagnose:
        diagnose_daily(daily)

    log.info("==== STEP 1+: Pipeline ====")
    results = run_pipeline(daily)

    log.info("==== FINAL SUMMARY ====")
    pm = results["primary_metrics"]
    mm = results["meta_metrics"]
    log.info(f"Primary: Rank IC={pm['rank_ic']:+.4f}, IC_IR={pm['ic_ir']:+.4f}, NDCG@5={pm['ndcg5']:.4f}")
    log.info(f"Meta:    AUC={mm['auc']:.4f}, hit_rate={mm['hit_rate']:.3f}, n_signals={mm['n_signals']}")
    log.info(f"Best threshold: {results['best_threshold']:.2f}")

    bt_p = results["bt_primary"]
    bt_m = results["bt_best"]
    ann = (252 / HORIZON_DAYS) ** 0.5   # правильная аннуализация под выбранный горизонт
    if not bt_p.empty:
        log.info(f"Primary-only backtest:")
        log.info(f"  Total return: {bt_p['equity'].iloc[-1] - 1:+.2%}")
        log.info(f"  Sharpe (annual): {bt_p['net'].mean() / (bt_p['net'].std() + 1e-9) * ann:.2f}")
        log.info(f"  MaxDD: {(bt_p['equity']/bt_p['equity'].cummax() - 1).min():+.2%}")
    if not bt_m.empty:
        log.info(f"With meta-filter (thr={results['best_threshold']:.2f}):")
        log.info(f"  Total return: {bt_m['equity'].iloc[-1] - 1:+.2%}")
        log.info(f"  Sharpe (annual): {bt_m['net'].mean() / (bt_m['net'].std() + 1e-9) * ann:.2f}")
        log.info(f"  MaxDD: {(bt_m['equity']/bt_m['equity'].cummax() - 1).min():+.2%}")

    # Save results for further inspection
    results["primary_oos"].to_parquet("primary_oos.parquet")
    results["meta_oos"].to_parquet("meta_oos.parquet")
    bt_p.to_csv("bt_primary.csv", index=False)
    bt_m.to_csv("bt_best.csv", index=False)
    log.info("Saved: primary_oos.parquet, meta_oos.parquet, bt_primary.csv, bt_best.csv")


if __name__ == "__main__":
    main()