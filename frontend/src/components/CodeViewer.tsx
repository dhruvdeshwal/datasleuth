import hljs from "highlight.js/lib/core";
import python from "highlight.js/lib/languages/python";
import sql from "highlight.js/lib/languages/sql";
import { useMemo, useState } from "react";

hljs.registerLanguage("sql", sql);
hljs.registerLanguage("python", python);

interface Props {
  code: string;
  language: "sql" | "pandas";
}

export function CodeViewer({ code, language }: Props) {
  const [copied, setCopied] = useState(false);
  const html = useMemo(
    () => hljs.highlight(code, { language: language === "sql" ? "sql" : "python" }).value,
    [code, language],
  );

  async function copy() {
    try {
      await navigator.clipboard.writeText(code);
    } catch {
      const area = document.createElement("textarea");
      area.value = code;
      document.body.appendChild(area);
      area.select();
      document.execCommand("copy");
      area.remove();
    }
    setCopied(true);
    setTimeout(() => setCopied(false), 1500);
  }

  return (
    <div className="overflow-hidden rounded-lg border border-line bg-surface-2">
      <div className="flex items-center justify-between border-b border-line px-3 py-1.5">
        <span className="text-xs font-medium uppercase tracking-wide text-ink-2">
          {language === "sql" ? "SQL" : "Pandas"}
        </span>
        <button
          type="button"
          onClick={copy}
          className="rounded px-2 py-1 text-xs font-medium text-accent hover:bg-accent/10"
        >
          {copied ? "Copied ✓" : `Copy ${language === "sql" ? "SQL" : "code"}`}
        </button>
      </div>
      <pre className="overflow-x-auto p-3 text-[13px] leading-relaxed">
        <code
          className="font-mono"
          aria-label={`${language} code`}
          dangerouslySetInnerHTML={{ __html: html }}
        />
      </pre>
    </div>
  );
}
