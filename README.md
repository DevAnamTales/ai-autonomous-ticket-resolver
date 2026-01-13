# 🤖 Autonomous AI Incident Resolver (Self-Healing IT Operations)

**Author:** Anam Rehman  
**GitHub:** https://github.com/DevAnamTales  
**Stack:** Python · Flask · LLM · AWS · ServiceNow · Vector DB · Automation

---

## 🚀 Overview

This project implements an **Autonomous AI Agent** capable of **triaging, diagnosing, and resolving ServiceNow incidents end-to-end without human intervention**.

It demonstrates a **self-healing IT operations architecture** where AI not only understands problems but also executes automated remediation actions and resolve tickets automatically.

---

## 🧠 Autonomous Self-Healing Use Case

Once an incident is created in ServiceNow:

1. The AI agent analyzes the ticket description  
2. Predicts the most likely root cause using LLM + Vector Search  
3. Executes automated remediation scripts  
4. Verifies recovery  
5. Updates and closes the ServiceNow ticket  
6. Escalates to humans only if risk or failure is detected  

This mimics real-world **OSS / ITSM autonomous operations.**

---

## 🏗 Architecture


**Flow:**

ServiceNow → Flask API → LLM Diagnose Chain → Autonomous Agent → Remediation Layer → ServiceNow


