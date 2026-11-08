---
title: 22l-6761_A2_Ops VisionGuard API
emoji: 🛡️
colorFrom: gray
colorTo: gray
sdk: docker
app_port: 8001
pinned: false
---

# VisionGuard model API (22l-6761_A2_Ops)

Classifies an uploaded image as AI-generated or real. Part of an MLOps course assignment.

- `GET /health` - status, model version, threshold (public)
- `POST /predict` - multipart field `image`, header `X-API-Key` required

Source: https://github.com/halfsleeper2119/22l-6761_A2_Ops
