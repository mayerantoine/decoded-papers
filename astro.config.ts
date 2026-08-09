import { defineConfig, envField } from "astro/config";
import tailwindcss from "@tailwindcss/vite";
import sitemap from "@astrojs/sitemap";
import remarkToc from "remark-toc";
import remarkCollapse from "remark-collapse";
import {
  transformerNotationDiff,
  transformerNotationHighlight,
  transformerNotationWordHighlight,
} from "@shikijs/transformers";
import { transformerFileName } from "./src/utils/transformers/fileName";
import { SITE } from "./src/config";

// Tags were consolidated to a controlled vocabulary (RAG, Agents, Multi-Agent,
// Evaluation, Deep Research, Public Health AI, Python). These map the retired
// tag slugs onto their closest surviving tag so old links don't 404.
const tagRedirects = Object.fromEntries(
  Object.entries({
    "agentic-ai": "agents",
    agent: "agents",
    "autonomous-agents": "agents",
    "agentic-patterns": "agents",
    "reflection-loop": "agents",
    "open-ai-agents-sdk": "agents",
    "re-act": "agents",
    anthropic: "agents",
    claude: "agents",
    llm: "agents",
    "deep-research-agent": "deep-research",
    "ai-research-tools": "deep-research",
    "evidence-synthesis": "deep-research",
    "rag-evaluation": "evaluation",
    "evaluation-dataset": "evaluation",
    ragas: "evaluation",
    "context-precision": "evaluation",
    faithfulness: "evaluation",
    "pub-med-qa": "evaluation",
    "medical-ai": "public-health-ai",
    "retrieval-augmented-generation-rag": "rag",
    "multi-document-summarization": "rag",
    "hybrid-retrieval": "rag",
    "chroma-db": "rag",
    "col-bert": "rag",
    "bm-25": "rag",
    specter: "rag",
    "x-sum": "rag",
    "lit-llm": "rag",
    "paper-qa": "rag",
    "debate-prompting": "rag",
    "conversational-recommendation": "agents",
    "recommender-system": "agents",
  }).map(([from, to]) => [`/tags/${from}`, `/tags/${to}`])
);

// https://astro.build/config
export default defineConfig({
  site: SITE.website,
  redirects: tagRedirects,
  integrations: [
    sitemap({
      filter: page => SITE.showArchives || !page.endsWith("/archives"),
    }),
  ],
  markdown: {
    remarkPlugins: [remarkToc, [remarkCollapse, { test: "Table of contents" }]],
    shikiConfig: {
      // For more themes, visit https://shiki.style/themes
      themes: { light: "github-dark", dark: "night-owl" },
      defaultColor: false,
      wrap: false,
      transformers: [
        transformerFileName({ style: "v2", hideDot: false }),
        transformerNotationHighlight(),
        transformerNotationWordHighlight(),
        transformerNotationDiff({ matchAlgorithm: "v3" }),
      ],
    },
  },
  vite: {
    // eslint-disable-next-line
    // @ts-ignore
    // This will be fixed in Astro 6 with Vite 7 support
    // See: https://github.com/withastro/astro/issues/14030
    plugins: [tailwindcss()],
    optimizeDeps: {
      exclude: ["@resvg/resvg-js"],
    },
  },
  image: {
    responsiveStyles: true,
    layout: "constrained",
  },
  env: {
    schema: {
      PUBLIC_GOOGLE_SITE_VERIFICATION: envField.string({
        access: "public",
        context: "client",
        optional: true,
      }),
    },
  },
  experimental: {
    preserveScriptOrder: true,
  },
});
