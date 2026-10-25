import { useEffect, useLayoutEffect, useRef, useState } from "react";
import {
  AreaSeries,
  CandlestickSeries,
  ColorType,
  CrosshairMode,
  HistogramSeries,
  LineSeries,
  createChart,
  createSeriesMarkers,
} from "lightweight-charts";
import { chartColors } from "../../data/constants";
import { dateLabel, money, percent, price, timeLabel } from "../../utils/formatters";

const EMA_PERIOD = 20;

function emaLine(candles, period = EMA_PERIOD) {
  if (!candles.length || period <= 0) return [];
  const alpha = 2 / (period + 1);
  const points = [{ time: candles[0].time, value: candles[0].close }];
  for (let i = 1; i < candles.length; i += 1) {
    const value = alpha * candles[i].close + (1 - alpha) * points[i - 1].value;
    points.push({ time: candles[i].time, value });
  }
  return points;
}

function nearestBarIndex(candles, time) {
  let low = 0;
  let high = candles.length - 1;
  while (low <= high) {
    const mid = (low + high) >> 1;
    const value = candles[mid].time;
    if (value === time) return mid;
    if (value < time) low = mid + 1;
    else high = mid - 1;
  }
  if (high < 0) return 0;
  if (low >= candles.length) return candles.length - 1;
  return time - candles[high].time <= candles[low].time - time ? high : low;
}

function visibleRangeForTrade(candles, trade) {
  const last = candles.length - 1;
  if (last < 0) return null;

  const entry = nearestBarIndex(candles, trade.entryTime);
  const exit = nearestBarIndex(candles, trade.exitTime);
  const start = Math.min(entry, exit);
  const end = Math.max(entry, exit);
  const span = Math.max(end - start, 1);
  let windowBars = Math.round(span / 0.46);
  windowBars = Math.max(windowBars, span + 14, Math.min(22, last));
  windowBars = Math.min(Math.max(windowBars, span), last);

  const pad = Math.max(0, windowBars - span);
  let from = start - Math.round(pad * 0.62);
  let to = from + windowBars;
  if (from < 0) {
    to -= from;
    from = 0;
  }
  if (to > last) {
    from = Math.max(0, from - (to - last));
    to = last;
  }
  from = Math.min(from, start);
  to = Math.max(to, end);
  return { from: from - 0.4, to: to + 0.65 };
}

function pinChartInView(node) {
  if (!node) return;
  const nav = document.querySelector(".app-nav");
  const targetTop = (nav?.getBoundingClientRect().bottom ?? 0) + 12;
  const delta = node.getBoundingClientRect().top - targetTop;
  if (Math.abs(delta) < 4) return;

  const scroller = scrollParent(node);
  const scrollingRoot = document.scrollingElement || document.documentElement;
  if (scroller === scrollingRoot || scroller === document.body) {
    window.scrollTo({ top: Math.max(0, window.scrollY + delta), behavior: "auto" });
    return;
  }
  scroller.scrollTo({ top: Math.max(0, scroller.scrollTop + delta), behavior: "auto" });
}

function scrollParent(node) {
  let parent = node.parentElement;
  while (parent) {
    const overflowY = getComputedStyle(parent).overflowY;
    if ((overflowY === "auto" || overflowY === "scroll") && parent.scrollHeight > parent.clientHeight + 1) {
      return parent;
    }
    parent = parent.parentElement;
  }
  return document.scrollingElement || document.documentElement;
}

export function TradingCharts({ data, focusToken = 0, metrics, onSelectTrade, run, selectedTrade, theme }) {
  const priceRef = useRef(null);
  const equityRef = useRef(null);
  const drawdownRef = useRef(null);
  const overlayRef = useRef(null);
  const chartsRef = useRef(null);
  const [tooltip, setTooltip] = useState(null);
  const [drawing, setDrawing] = useState(null);
  const [framed, setFramed] = useState(false);
  const [tools, setTools] = useState({ crosshair: true, tradePath: true });
  const onSelectTradeRef = useRef(onSelectTrade);
  const framedRef = useRef(false);
  const seenDataRef = useRef(null);
  const appliedFocusToken = useRef(0);
  onSelectTradeRef.current = onSelectTrade;
  if (seenDataRef.current !== data) {
    seenDataRef.current = data;
    framedRef.current = false;
    if (framed) setFramed(false);
  }

  useEffect(() => {
    if (!priceRef.current || !equityRef.current || !drawdownRef.current) return undefined;
    const chartTheme = theme === "dark"
      ? {
          panel: "#101820",
          text: "#98a5b2",
          grid: "rgba(226, 236, 244, 0.045)",
          border: "rgba(226, 236, 244, 0.07)",
        }
      : {
          panel: "#fbfcfd",
          text: chartColors.muted,
          grid: chartColors.grid,
          border: "rgba(23, 33, 43, 0.07)",
        };
    let syncing = false;

    const baseOptions = {
      autoSize: true,
      layout: {
        background: { type: ColorType.Solid, color: chartTheme.panel },
        textColor: chartTheme.text,
        fontFamily: "IBM Plex Sans",
        fontSize: 11,
      },
      grid: {
        vertLines: { color: chartTheme.grid },
        horzLines: { color: chartTheme.grid },
      },
      rightPriceScale: {
        borderColor: chartTheme.border,
        scaleMargins: { top: 0.12, bottom: 0.18 },
      },
      timeScale: {
        borderColor: chartTheme.border,
        timeVisible: true,
        secondsVisible: false,
        minBarSpacing: 0.01,
        tickMarkFormatter: (time) => timeLabel(time),
      },
      crosshair: {
        mode: tools.crosshair ? CrosshairMode.Normal : CrosshairMode.Magnet,
        vertLine: { color: "rgba(15, 142, 168, 0.72)", style: 2, width: 1 },
        horzLine: { color: "rgba(15, 142, 168, 0.42)", style: 2, width: 1 },
      },
      handleScroll: true,
      handleScale: true,
    };

    const priceChart = createChart(priceRef.current, {
      ...baseOptions,
      localization: {
        priceFormatter: (value) => price(value),
        timeFormatter: (time) => `${dateLabel(time)} ET`,
      },
    });
    const candleSeries = priceChart.addSeries(CandlestickSeries, {
      upColor: chartColors.green,
      downColor: chartColors.red,
      wickUpColor: chartColors.green,
      wickDownColor: chartColors.red,
      borderVisible: false,
      priceFormat: { type: "price", precision: 2, minMove: 0.01 },
    });
    candleSeries.setData(data.candles);

    const volumeSeries = priceChart.addSeries(HistogramSeries, {
      priceScaleId: "",
      priceFormat: { type: "volume" },
      base: 0,
    });
    volumeSeries.setData(data.volume);
    volumeSeries.priceScale().applyOptions({ scaleMargins: { top: 0.82, bottom: 0 } });

    const emaPoints = emaLine(data.candles);
    const emaByTime = new Map(emaPoints.map((point) => [point.time, point.value]));
    const emaSeries = priceChart.addSeries(LineSeries, {
      color: theme === "dark" ? "#e0a82e" : "#c47d00",
      lineWidth: 2,
      priceLineVisible: false,
      lastValueVisible: false,
      crosshairMarkerVisible: true,
      crosshairMarkerRadius: 3,
      priceFormat: { type: "price", precision: 2, minMove: 0.01 },
    });
    emaSeries.setData(emaPoints);

    createSeriesMarkers(
      candleSeries,
      data.trades.flatMap((trade) => [
        {
          time: trade.entryTime,
          position: trade.side === "Long" ? "belowBar" : "aboveBar",
          color: trade.side === "Long" ? chartColors.green : chartColors.red,
          shape: trade.side === "Long" ? "arrowUp" : "arrowDown",
          text: trade.side,
        },
        {
          time: trade.exitTime,
          position: trade.side === "Long" ? "aboveBar" : "belowBar",
          color: trade.pnl >= 0 ? chartColors.green : chartColors.red,
          shape: "circle",
          text: "Exit",
        },
      ]),
    );

    const equityChart = createChart(equityRef.current, {
      ...baseOptions,
      height: 110,
      rightPriceScale: {
        borderColor: chartTheme.border,
        scaleMargins: { top: 0.15, bottom: 0.12 },
      },
    });
    const equitySeries = equityChart.addSeries(AreaSeries, {
      lineColor: chartColors.green,
      topColor: chartColors.greenSoft,
      bottomColor: "rgba(20, 154, 90, 0.02)",
      lineWidth: 2,
      priceFormat: { type: "price", precision: 0, minMove: 1 },
    });
    equitySeries.setData(data.equity);

    const drawdownChart = createChart(drawdownRef.current, {
      ...baseOptions,
      height: 94,
      rightPriceScale: {
        borderColor: chartTheme.border,
        scaleMargins: { top: 0.12, bottom: 0.18 },
      },
      localization: { priceFormatter: (value) => `${value.toFixed(0)}%` },
    });
    const drawdownSeries = drawdownChart.addSeries(AreaSeries, {
      lineColor: chartColors.red,
      topColor: "rgba(200, 63, 58, 0.02)",
      bottomColor: chartColors.redSoft,
      lineWidth: 2,
      priceFormat: { type: "price", precision: 1, minMove: 0.1 },
    });
    drawdownSeries.setData(data.drawdown);

    const updateDrawing = () => {
      const trade = selectedTradeRef.current;
      if (!trade || !overlayRef.current) return;
      const x1 = priceChart.timeScale().timeToCoordinate(trade.entryTime);
      const x2 = priceChart.timeScale().timeToCoordinate(trade.exitTime);
      const y1 = candleSeries.priceToCoordinate(trade.entryPrice);
      const y2 = candleSeries.priceToCoordinate(trade.exitPrice);
      if ([x1, x2, y1, y2].some((point) => point === null)) {
        setDrawing(null);
        return;
      }

      const left = Math.min(x1, x2);
      const top = Math.min(y1, y2);
      const width = Math.max(1, Math.abs(x2 - x1));
      const height = Math.max(1, Math.abs(y2 - y1));
      const length = Math.sqrt(width ** 2 + height ** 2);
      const angle = (Math.atan2(y2 - y1, x2 - x1) * 180) / Math.PI;

      setDrawing({
        region: {
          left,
          top,
          width,
          height,
          borderColor: trade.pnl >= 0 ? chartColors.green : chartColors.red,
          background: trade.pnl >= 0 ? chartColors.greenSoft : chartColors.redSoft,
        },
        line: {
          left: x1,
          top: y1,
          width: length,
          transform: `rotate(${angle}deg)`,
          background: trade.pnl >= 0 ? chartColors.cyan : chartColors.red,
        },
        entry: { left: x1, top: y1, label: `Entry ${price(trade.entryPrice)}` },
        exit: { left: x2, top: y2, label: `Exit ${price(trade.exitPrice)}` },
      });
    };

    const selectedTradeRef = { current: selectedTrade };
    const focusTrade = (trade, { pin = false } = {}) => {
      const range = visibleRangeForTrade(data.candles, trade);
      if (!range) return;
      selectedTradeRef.current = trade;
      candleSeries.priceScale().setAutoScale(true);
      priceChart.timeScale().setVisibleLogicalRange(range);
      updateDrawing();
      if (pin) pinChartInView(priceRef.current?.closest(".chart-stack"));
      requestAnimationFrame(updateDrawing);
    };
    const syncRange = (source, targets) => {
      source.timeScale().subscribeVisibleLogicalRangeChange((range) => {
        if (!range || syncing) return;
        syncing = true;
        targets.forEach((chart) => chart.timeScale().setVisibleLogicalRange(range));
        syncing = false;
        requestAnimationFrame(updateDrawing);
      });
    };

    chartsRef.current = {
      priceChart,
      equityChart,
      drawdownChart,
      candleSeries,
      focusTrade,
      updateDrawing: () => updateDrawing(),
      selectedTradeRef,
    };

    syncRange(priceChart, [equityChart, drawdownChart]);
    syncRange(equityChart, [priceChart, drawdownChart]);
    syncRange(drawdownChart, [priceChart, equityChart]);

    if (framedRef.current && selectedTradeRef.current) {
      focusTrade(selectedTradeRef.current);
    } else {
      priceChart.timeScale().fitContent();
      equityChart.timeScale().fitContent();
      drawdownChart.timeScale().fitContent();
    }

    priceChart.subscribeClick((param) => {
      if (!param.time) return;
      const nearest = data.trades.reduce((best, trade) => {
        const distance = Math.min(Math.abs(trade.entryTime - param.time), Math.abs(trade.exitTime - param.time));
        return distance < best.distance ? { trade, distance } : best;
      }, { trade: selectedTradeRef.current, distance: Number.POSITIVE_INFINITY });
      if (nearest.distance < 1800 && nearest.trade) onSelectTradeRef.current?.(nearest.trade);
    });

    priceChart.subscribeCrosshairMove((param) => {
      if (!param.point || !param.time || !priceRef.current) {
        setTooltip(null);
        return;
      }
      const candle = data.candles.find((item) => item.time === param.time);
      if (!candle) return;
      const hoverTrade = data.trades.find((trade) => param.time >= trade.entryTime && param.time <= trade.exitTime);
      setTooltip({
        x: Math.min(param.point.x + 18, priceRef.current.clientWidth - 220),
        y: Math.max(param.point.y - 16, 16),
        candle,
        ema: emaByTime.get(candle.time),
        trade: hoverTrade,
      });
    });

    const resizeObserver = new ResizeObserver(() => updateDrawing());
    resizeObserver.observe(priceRef.current);
    window.addEventListener("resize", updateDrawing);
    requestAnimationFrame(updateDrawing);

    return () => {
      resizeObserver.disconnect();
      window.removeEventListener("resize", updateDrawing);
      priceChart.remove();
      equityChart.remove();
      drawdownChart.remove();
      chartsRef.current = null;
    };
  }, [data, theme, tools.crosshair]);

  useLayoutEffect(() => {
    const charts = chartsRef.current;
    if (!charts) return;
    charts.selectedTradeRef.current = selectedTrade;
    if (!selectedTrade || !focusToken || focusToken === appliedFocusToken.current) {
      charts.updateDrawing();
      return;
    }
    appliedFocusToken.current = focusToken;
    framedRef.current = true;
    setFramed(true);
    charts.focusTrade(selectedTrade, { pin: true });
  }, [focusToken, selectedTrade]);

  const fitAll = () => {
    const charts = chartsRef.current;
    framedRef.current = false;
    setFramed(false);
    if (!charts) return;
    charts.candleSeries.priceScale().setAutoScale(true);
    charts.priceChart.timeScale().fitContent();
  };

  const lastCandle = data.candles.at(-1);
  const priorCandle = data.candles.at(-2);
  const finalEquity = data.equity.at(-1)?.value;
  const currentDrawdown = data.drawdown.at(-1)?.value;
  const maxDrawdown = data.drawdown.reduce((lowest, point) => Math.min(lowest, point.value), 0);
  const change = lastCandle && priorCandle ? lastCandle.close - priorCandle.close : 0;
  const changePct = priorCandle ? change / priorCandle.close : 0;

  return (
    <section className={framed ? "chart-stack framed" : "chart-stack"} aria-label="Strategy chart workspace">
      <div className="chart-toolbar">
        <div>
          <span className="chart-title">{run?.market ?? "-"} {run?.timeframe ?? "-"}</span>
          {framed && selectedTrade ? (
            <>
              <span className="chart-focus">Trade #{selectedTrade.id}</span>
              <span className="ohlc">{selectedTrade.side} {price(selectedTrade.entryPrice)} → {price(selectedTrade.exitPrice)}</span>
              <span className={selectedTrade.pnlClass}>{money(selectedTrade.pnl)} · {selectedTrade.r}R</span>
            </>
          ) : lastCandle && (
            <>
              <span className="ohlc">O {price(lastCandle.open)}</span>
              <span className="ohlc">H {price(lastCandle.high)}</span>
              <span className="ohlc">L {price(lastCandle.low)}</span>
              <span className="ohlc">C {price(lastCandle.close)}</span>
              <span className={change >= 0 ? "positive" : "negative"}>{price(change)} ({percent(changePct)})</span>
            </>
          )}
        </div>
        <div className="tool-group" aria-label="Chart tools">
          <span className="ema-legend">EMA 20</span>
          <button type="button" onClick={fitAll}>Fit</button>
          {["crosshair", "tradePath"].map((tool) => (
            <button
              className={tools[tool] ? "active" : ""}
              key={tool}
              type="button"
              onClick={() => setTools((current) => ({ ...current, [tool]: !current[tool] }))}
            >
              {tool === "tradePath" ? "Trade path" : tool.charAt(0).toUpperCase() + tool.slice(1)}
            </button>
          ))}
        </div>
      </div>
      <div className="price-chart-wrap">
        <div className="price-chart" ref={priceRef} />
        <div className="drawing-layer" ref={overlayRef}>
          {drawing && tools.tradePath && (
            <>
              <div className="trade-region" style={drawing.region} />
              <div className="trade-line" style={drawing.line} />
              <div className="trade-point entry-point" style={drawing.entry} />
              <div className="trade-point exit-point" style={drawing.exit} />
              <div className="trade-label entry-label" style={drawing.entry}>{drawing.entry.label}</div>
              <div className="trade-label exit-label" style={drawing.exit}>{drawing.exit.label}</div>
            </>
          )}
        </div>
        {tooltip && (
          <div className="chart-tooltip" style={{ left: tooltip.x, top: tooltip.y }}>
            <div className="tooltip-date">{dateLabel(tooltip.candle.time)}</div>
            <div><span>Open</span><b>{price(tooltip.candle.open)}</b></div>
            <div><span>High</span><b>{price(tooltip.candle.high)}</b></div>
            <div><span>Low</span><b>{price(tooltip.candle.low)}</b></div>
            <div><span>Close</span><b>{price(tooltip.candle.close)}</b></div>
            {tooltip.ema != null && <div><span>EMA 20</span><b>{price(tooltip.ema)}</b></div>}
            {tooltip.trade && (
              <div className="tooltip-trade">
                <strong>Trade {tooltip.trade.id}</strong>
                <span className={tooltip.trade.pnlClass}>{money(tooltip.trade.pnl)} · {tooltip.trade.r}R</span>
              </div>
            )}
          </div>
        )}
      </div>
      <div className="subchart">
        <div className="panel-label">
          Equity curve
          {finalEquity !== undefined && <span>Final: {money(finalEquity)}</span>}
          <strong>Net {percent(metrics?.total_return ?? 0)}</strong>
        </div>
        <div className="subchart-canvas" ref={equityRef} />
      </div>
      <div className="subchart compact">
        <div className="panel-label">
          Drawdown
          <span>Max: {maxDrawdown.toFixed(2)}%</span>
          <strong className="negative">Current {(currentDrawdown ?? 0).toFixed(2)}%</strong>
        </div>
        <div className="subchart-canvas" ref={drawdownRef} />
      </div>
    </section>
  );
}
