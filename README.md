# Blackjack CV - Computer Vision Card Counting Assistant

A real-time computer vision system for detecting playing cards and providing optimal blackjack strategy recommendations through a wearable device.

## Overview

This project uses YOLOv8 for playing card detection and implements card counting strategies (Hi-Lo, KO, etc.) and basic blackjack strategy to provide real-time gameplay assistance. Designed to run on an Orange Pi 5 with wearable camera module.

## Features (Planned)

- ✅ Real-time playing card detection using YOLOv8
- 🚧 Player vs dealer card differentiation
- 🚧 Multiple card counting systems
- 🚧 Basic strategy recommendations
- 🚧 Haptic feedback for discreet notifications
- 🚧 Web interface for configuration and statistics
- 🚧 Orange Pi 5 NPU acceleration

## Hardware Requirements

- **Development**: Any computer with webcam
- **Deployment**: Orange Pi 5 (8GB recommended)
- **Camera**: USB webcam or CSI camera module
- **Feedback**: Haptic motor (optional)

## Software Requirements

- Python 3.11+
- CUDA-capable GPU (optional, for faster development)
- Webcam for testing

## Installation

1. Clone the repository:

```bash
git clone https://github.com/yourusername/blackjack-cv.git
cd blackjack-cv
```
