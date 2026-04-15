# IE 4198 Midterm Project Plan
## Deep Learning for Supply Chain Forecasting

**Project:** A Comparative Analysis of Classical and Deep Learning Forecasting for Supply Chain Inventory Optimization  
**Date:** April 15, 2026 (Midterm)  
**Deadline:** Friday, April 17, 2026 (Midterm Report Submission)  
**Supervisor:** Prof. Dr. Serol Bulkan

---

## 1. Executive Summary

This document outlines the comprehensive plan to complete the M5 Walmart Capstone Project for IE 4198. The project builds upon the IE 4197 foundation, which established:
- A rigorous 10-algorithm benchmarking framework
- Financial impact analysis showing 19.5% cost savings with LightGBM
- A preliminary Next.js Decision Support System dashboard

The IE 4198 phase focuses on **advanced modeling** (DeepAR implementation) and **deployment** of the complete decision support system.

---

## 2. Current Project Status (IE 4197 Completed)

### 2.1. Infrastructure
- [x] Hybrid monorepo (Python backend + Next.js frontend)
- [x] Polars-based data pipeline for 59M rows
- [x] M5 dataset ingestion and preprocessing
- [x] Store CA_1 subset (3,049 items) for pilot analysis

### 2.2. Modeling (Phase I Complete)
| Model | RMSE | Total Cost | Savings vs Naive |
|-------|------|------------|------------------|
| **LightGBM** | 2.10 | $515,513 | **+19.5%** |
| Holt-Winters | 2.31 | $529,096 | +17.4% |
| Naive (baseline) | 2.86 | $640,703 | - |
| LSTM | 3.58 | $1,229,646 | -91% |

### 2.3. Deliverables (IE 4197)
- [x] Algorithm "Drag Race" benchmark report
- [x] Inventory optimization simulation script
- [x] Preliminary frontend dashboard
- [x] Final Report Template (docs/Final_Report_Template.md)

---

## 3. IE 4198 Objectives (Spring 2026)

### 3.1. Primary Goals
1. **Advanced Probabilistic Modeling:** Implement DeepAR to address the "Sparsity Penalty" observed in LSTM
2. **Decision Support System:** Complete the Next.js dashboard with real-time inventory simulation
3. **Sensitivity Analysis:** Analyze cost parameter sensitivity (Holding vs Stockout ratios)
4. **Full-Scale Execution:** Extend training to the complete 30,490-item dataset

### 3.2. Secondary Goals
- Academic paper-style documentation
- Presentation-ready visualizations
- Reproducible pipeline for future students

---

## 4. Detailed Timeline (Midterm to Final)

### Phase A: Midterm Preparation (Week 1: April 13-17, 2026)

| Task | Description | Status | Deadline |
|------|-------------|--------|----------|
| A.1 | Review existing code and validate changes | Complete | April 14 |
| A.2 | Update benchmark results if needed | Complete | April 14 |
| A.3 | Write midterm progress report | **In Progress** | April 16 |
| A.4 | Finalize IE 4198 Midterm Report | **In Progress** | **April 17 (Friday)** |

### Phase B: DeepAR Implementation (Week 2-3: April 20-May 1, 2026)

| Task | Description | Status | Deadline |
|------|-------------|--------|----------|
| B.1 | Complete DeepAR training pipeline | In Progress | April 25 |
| B.2 | Validate DeepAR on validation set | Pending | April 27 |
| B.3 | Update financial impact analysis | Pending | April 29 |
| B.4 | Compare DeepAR vs LightGBM results | Pending | May 1 |

### Phase C: Dashboard Enhancement (Week 3-4: April 27-May 8, 2026)

| Task | Description | Status | Deadline |
|------|-------------|--------|----------|
| C.1 | Add DeepAR forecasts to frontend | Pending | May 3 |
| C.2 | Implement "What-If" scenario simulator | Pending | May 5 |
| C.3 | Add sensitivity analysis visualization | Pending | May 7 |
| C.4 | Polish UI and add team info | Pending | May 8 |

### Phase D: Final Report & Defense (Week 5-6: May 11-June 2026)

| Task | Description | Status | Deadline |
|------|-------------|--------|----------|
| D.1 | Write complete IE 4198 Final Report | Pending | May 15 |
| D.2 | Compile all appendices and code | Pending | May 18 |
| D.3 | Prepare presentation slides | Pending | May 25 |
| D.4 | Project defense | Pending | June 2026 |

---

## 5. Technical Tasks for Midterm Report

### 5.1. Required Updates
1. **Update RMSE values:** Use consistent metrics between benchmark and simulation
2. **Document DeepAR progress:** Explain the "Sparsity Penalty" solution approach
3. **Dashboard screenshots:** Include current state of the Next.js application
4. **Gantt Chart:** Visual timeline for remaining work

### 5.2. Code Validation Checklist
- [x] `backend/scripts/training/train_deepar.py` - DeepAR implementation
- [x] `backend/scripts/data_preparation/export_frontend_data.py` - Data export
- [x] `frontend/src/app/page.tsx` - Interactive dashboard
- [x] `frontend/public/data/summary.json` - Benchmark results
- [x] `frontend/public/data/sku_data.json` - SKU-level forecasts

### 5.3. Potential Issues to Address
1. **LSTM underperformance:** Addressed via DeepAR (Poisson likelihood for zero-inflated data)
2. **MPS/GPU instability:** Fixed by forcing CPU training in train_deepar.py
3. **Data leakage:** Verified with proper train/validation split

---

## 6. Risk Assessment

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| DeepAR training time | High | Medium | Use CA_1 subset first, then scale up |
| Next.js deployment issues | Medium | Low | Use local dev server for testing |
| Incomplete sensitivity analysis | Medium | Medium | Simplify to single parameter (z-score) |
| Report formatting issues | Low | High | Use provided Word template |

---

## 7. Midterm Deliverables (Due Friday, April 17)

1. **IE 4198 Midterm Progress Report** (3-5 pages)
   - Executive summary of progress
   - Updated benchmark results
   - Dashboard status
   - Work plan for remaining tasks

2. **Code Deliverables**
   - All scripts in `backend/scripts/`
   - Frontend dashboard functional at `localhost:3000`

3. **Presentation Materials** (optional)
   - 5-10 minute overview for midterm meeting

---

## 8. Next Steps After Midterm

1. Complete DeepAR training and validation
2. Add DeepAR results to frontend dashboard
3. Perform sensitivity analysis on H/S cost ratios
4. Write comprehensive final report following EK-7 template
5. Prepare for project defense

---

**Document Prepared:** April 15, 2026  
**Last Updated:** April 15, 2026
