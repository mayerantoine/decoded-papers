---
author: Mayer Antoine
pubDatetime: 2025-10-18
modDatetime: 2025-10-18
title: Implementing Lit-LLM An Agentic Workflow for Synthesizing Literature Reviews
slug: implementing-lit-llm-an-agentic-workflow-for-synthesizing-litterature-reviews
draft: True
tags:
  - Retrieval-augmented generation(RAG)
  - Agent
description: Implementing Lit-LLM An Agentic Workflow for Synthesizing Research Paper Literature Reviews using OpenAI Agents. We reproduced and adapted the multi-document summarization approach from the paper “LitLLMs, LLMs for Literature Review Are we there yet?” (2024) for our own use and testing.
---

## Table of contents

## Agentic Workflow

Implementing Lit-LLM An Agentic Workflow for Synthesizing Research Paper Literature Reviews using OpenAI Agents. We reproduced and adapted the multi-document summarization approach from the paper "LitLLMs, LLMs for Literature Review Are we there yet?" (2024) for our own use and testing.

The general workflow starts with a paper idea and research questions. You then use keywords to search several databases and download abstracts related to background work, previous research, and gaps connected to your research question.

Given your research idea and list of abstracts, this tool quickly generates comprehensive "Related Work" sections by intelligently retrieving, scoring, and synthesizing insights from your abstracts.