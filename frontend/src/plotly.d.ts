// plotly.js ships partial bundles without type declarations.
declare module "plotly.js/lib/core" {
  import type * as PlotlyTypes from "plotly.js";
  const Plotly: typeof PlotlyTypes & { register(modules: unknown[]): void };
  export default Plotly;
}
declare module "plotly.js/lib/bar";
declare module "plotly.js/lib/box";
declare module "plotly.js/lib/histogram";
declare module "plotly.js/lib/pie";
declare module "plotly.js/lib/scatter";
