import type { Data, Layout, PlotlyHTMLElement } from "plotly.js";
import PlotlyModule from "plotly.js/lib/core";
import barModule from "plotly.js/lib/bar";
import boxModule from "plotly.js/lib/box";
import histogramModule from "plotly.js/lib/histogram";
import pieModule from "plotly.js/lib/pie";
import scatterModule from "plotly.js/lib/scatter";
import { useMemo, useRef } from "react";
import factoryModule from "react-plotly.js/factory";

// These packages are CommonJS; depending on the bundler the default import is either the
// export itself or a `{ default }` wrapper. Normalise both.
function interop<T>(mod: T): T {
  return ((mod as unknown as { default?: T }).default ?? mod) as T;
}

const Plotly = interop(PlotlyModule);
const createPlotlyComponent = interop(factoryModule);
Plotly.register([barModule, boxModule, histogramModule, pieModule, scatterModule].map(interop));
const Plot = createPlotlyComponent(Plotly);

interface Props {
  chartJson: string;
  dark: boolean;
  filename?: string;
}

const INK = { light: "#52514e", dark: "#c3c2b7" };
const GRID = { light: "#e1e0d9", dark: "#2c2c2a" };
const SURFACE = { light: "#fcfcfb", dark: "#1a1a19" };
// Validated categorical palette: light-surface steps (as rendered by the server) and dark-surface steps.
const LIGHT_COLORWAY = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"];
const DARK_COLORWAY = ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#008300", "#9085e9", "#e66767"];

export function ChartView({ chartJson, dark, filename = "chart" }: Props) {
  const graph = useRef<PlotlyHTMLElement | null>(null);
  const figure = useMemo(() => {
    // Trace colors are baked into the figure JSON (light palette); swap in the dark-surface steps.
    const themed = dark
      ? LIGHT_COLORWAY.reduce((json, hex, i) => json.split(hex).join(DARK_COLORWAY[i]), chartJson)
      : chartJson;
    const parsed = JSON.parse(themed) as { data: Data[]; layout: Partial<Layout> };
    const mode = dark ? "dark" : "light";
    const axis = { gridcolor: GRID[mode], linecolor: GRID[mode], color: INK[mode] };
    const layout: Partial<Layout> = {
      ...parsed.layout,
      autosize: true,
      paper_bgcolor: SURFACE[mode],
      plot_bgcolor: SURFACE[mode],
      font: { ...(parsed.layout.font ?? {}), color: INK[mode] },
      title: { ...(parsed.layout.title as object), font: { size: 16, color: dark ? "#ffffff" : "#0b0b0b" } },
      xaxis: { ...(parsed.layout.xaxis ?? {}), ...axis },
      yaxis: { ...(parsed.layout.yaxis ?? {}), ...axis },
      ...(dark ? { colorway: DARK_COLORWAY } : {}),
    };
    return { data: parsed.data, layout };
  }, [chartJson, dark]);

  async function downloadPng() {
    if (graph.current) await Plotly.downloadImage(graph.current, { format: "png", filename, width: 900, height: 520 });
  }

  return (
    <figure className="rounded-lg border border-line bg-surface p-2" aria-label="Chart of the result">
      <Plot
        data={figure.data}
        layout={figure.layout}
        config={{ displaylogo: false, responsive: true }}
        useResizeHandler
        style={{ width: "100%", height: 360 }}
        onInitialized={(_f, div) => (graph.current = div as PlotlyHTMLElement)}
        onUpdate={(_f, div) => (graph.current = div as PlotlyHTMLElement)}
      />
      <figcaption className="flex justify-end px-1 pt-1">
        <button
          type="button"
          onClick={downloadPng}
          className="rounded px-2 py-1 text-xs font-medium text-accent hover:bg-accent/10"
        >
          Download PNG
        </button>
      </figcaption>
    </figure>
  );
}
