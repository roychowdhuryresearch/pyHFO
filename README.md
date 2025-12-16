# PyHFO
<img src="src/ui/images/icon1.png" alt="pyHFO logo" style="width:30%;">

[![License](https://img.shields.io/badge/License-UCLA%20Academic-blue.svg)](LICENSE.txt)
[![Python](https://img.shields.io/badge/Python-3.9-blue.svg)](https://www.python.org/downloads/)
[![Latest Release](https://img.shields.io/github/v/release/roychowdhuryresearch/pyHFO)](https://github.com/roychowdhuryresearch/pyHFO/releases)

[ProjectPage](https://roychowdhuryresearch.github.io/PyHFO_Project_Page/) |
[Download](https://github.com/roychowdhuryresearch/pyHFO/releases) | 
[Manual](https://docs.google.com/document/d/1KzQpfuPFDk2lr9V3TgSkmc21jISB54ZOxo3pIYKQsp0/edit?usp=sharing)

PyHFO is a multi-window desktop application providing an integrated and user-friendly platform that includes time-efficient HFO detection algorithms such as short-term energy (STE) and Montreal Neurological Institute and Hospital (MNI) detectors and deep learning models for artifact and HFO with spike classification.

## Table of Contents
- [Features](#features)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage](#usage)
- [Troubleshooting](#troubleshooting)
- [Citation](#citation)
- [Related Projects](#related-projects)
- [License](#license)
- [Contact & Support](#contact--support)
- [Acknowledgments](#acknowledgments)

## Features

- **Multiple Detection Algorithms**: Implements both traditional (STE, MNI) and deep learning-based HFO detection methods
- **Real-time Visualization**: Multi-window interface for simultaneous viewing of EEG signals and detected events
- **AI-Powered Classification**: Deep learning models for distinguishing HFOs from artifacts and identifying HFOs with spikes
- **EDF File Support**: Compatible with standard EDF/EDF+ formats commonly used in clinical neurophysiology
- **Interactive Annotation**: Manual review and editing capabilities for detected HFOs
- **Batch Processing**: Analyze multiple files efficiently with consistent parameters
- **Export Capabilities**: Save detection results and annotations for further analysis

## Prerequisites

- **Operating System**: macOS, Windows, or Linux
- **Python**: Version 3.9 (recommended)
- **Memory**: Minimum 8GB RAM recommended
- **Display**: Multi-monitor setup recommended for optimal workflow

## Citation

If you find our project useful in your research, please cite:

```bibtex
@article{ding2025pyhfo,
  title={PyHFO 2.0: An open-source platform for deep learning–based clinical high-frequency oscillations analysis},
  author={Ding, Y. and Zhang, Y. and Duan, C. and Daida, A. and Zhang, Y. and Kanai, S. and Lu, M. and Hussain, S. and Staba, R. J. and Nariai, H. and Roychowdhury, V.},
  journal={Journal of Neural Engineering},
  volume={22},
  number={5},
  pages={056040},
  year={2025},
  doi={10.1088/1741-2552/ae10e0}
}

@article{zhang2024pyhfo,
  title={PyHFO: lightweight deep learning-powered end-to-end high-frequency oscillations analysis application},
  author={Zhang, Y. and Liu, L. and Ding, Y. and Chen, X. and Monsoor, T. and Daida, A. and Oana, S. and Hussain, S. A. and Sankar, R. and Fallah, A. and Santana-Gomez, C. and Engel, J. and Staba, R. J. and Speier, W. and Zhang, J. and Nariai, H. and Roychowdhury, V.},
  journal={Journal of Neural Engineering},
  year={2024},
  doi={10.1088/1741-2552/ad4916}
}
```

## Related Projects

* [HFODetector](https://github.com/roychowdhuryresearch/HFO_Detector) - A Python toolbox for very fast HFO detection.

* [HFO-Classification](https://github.com/roychowdhuryresearch/HFO-Classification) - Many HFO classification projects powered by deep learning.

* [EEG-Viz](https://github.com/jebbica/EEG-Viz) - A Python toolbox for EEG visualization.


## Installation

### Option 1: Standalone Application (Recommended for End Users)

You can download the latest version of PyHFO from the [releases](https://github.com/roychowdhuryresearch/pyHFO/releases) page.

#### macOS Users

If you choose to use the **macOS version** of the standalone distributable application, please follow these steps:

1. **Download and unzip** the `.zip` file.
2. You will get a file named `pyHFO.dmg`.
3. Navigate to the directory containing the `pyHFO.dmg` file.
4. Open the terminal and run the following command to remove the quarantine attribute (required due to macOS security settings for applications downloaded from the internet):

```bash
xattr -cr pyHFO.dmg
```

5. Double-click `pyHFO.dmg` to mount it and drag PyHFO to your Applications folder.

### Option 2: Install from Source (For Developers)

```bash
git clone https://github.com/roychowdhuryresearch/pyHFO.git 
cd pyHFO
conda create -n pyhfo python=3.9
conda activate pyhfo
pip install -r requirements.txt
python main.py
```

## Quick Start

1. **Launch PyHFO**: Run the application (standalone or via `python main.py`)
2. **Load EDF File**: Click "File" → "Open" and select your EEG data file
3. **Select Channels**: Choose the channels you want to analyze
4. **Configure Detection**: Select detection method (STE, MNI, or Deep Learning) and adjust parameters
5. **Run Detection**: Click "Detect" to start HFO detection
6. **Review Results**: Use the interactive interface to review and annotate detected HFOs
7. **Export**: Save your results for further analysis

## Usage

PyHFO provides an intuitive multi-window interface for HFO detection and analysis. The main workflow includes:

1. **Data Loading**: Import EEG data in EDF/EDF+ format
2. **Channel Selection**: Choose channels of interest for analysis
3. **Detection Configuration**: Select and configure detection algorithms
4. **Detection Execution**: Run automated HFO detection
5. **Visual Review**: Examine detected events in the interactive waveform viewer
6. **Manual Annotation**: Add, edit, or remove detections as needed
7. **Result Export**: Save detection results and annotations

The overview of the PyHFO interface is shown below:
![PyHFO Interface Overview](img/overview1.png)

For detailed instructions and advanced features, please refer to the comprehensive [user manual](https://docs.google.com/document/d/1KzQpfuPFDk2lr9V3TgSkmc21jISB54ZOxo3pIYKQsp0/edit?usp=sharing).

## Troubleshooting

### PyQt Installation Issues

If you encounter failures when installing PyQt via `requirements.txt`:

```bash
pip install pyqt5
```

Then edit `requirements.txt` to remove or comment out PyQt-related lines before running `pip install -r requirements.txt` again.

### macOS Security Warning

If macOS prevents you from opening the application with a security warning, use the `xattr -cr` command as described in the [Installation](#installation) section.

### Memory Issues

For large EEG files, ensure you have sufficient RAM available. Close other applications if you experience performance issues.

### Missing Dependencies

If you encounter import errors, verify all dependencies are installed:

```bash
pip install -r requirements.txt --force-reinstall
```

## License

This project is licensed under the UCLA Academic License - see the [LICENSE.txt](LICENSE.txt) file for details.

## Contact & Support

- **Issues & Bug Reports**: [GitHub Issues](https://github.com/roychowdhuryresearch/pyHFO/issues)
- **Questions & Discussions**: [GitHub Discussions](https://github.com/roychowdhuryresearch/pyHFO/discussions)
- **Documentation**: [User Manual](https://docs.google.com/document/d/1KzQpfuPFDk2lr9V3TgSkmc21jISB54ZOxo3pIYKQsp0/edit?usp=sharing)
- **Project Website**: [PyHFO Project Page](https://roychowdhuryresearch.github.io/PyHFO_Project_Page/)

For academic collaborations or research inquiries, please contact Prof. Vwani Roychowdhury through the UCLA ECE department.

## Acknowledgments

### Contributors:
This project is under supervision of Prof. [Vwani Roychowdhury](https://www.ee.ucla.edu/vwani-p-roychowdhury/).

Department of Electrical and Computer Engineering, University of California, Los Angeles
- [Yipeng Zhang](https://zyp5511.github.io/)
- [Lawrence Liu](https://www.linkedin.com/in/lawrence-liu-0a01391a7/)
- [Yuanyi Ding](https://www.linkedin.com/in/yuanyi-ding-4a981a132/)
- [Xin Chen](https://www.linkedin.com/in/xin-chen-980521/)
- [Jessica Lin](https://www.linkedin.com/in/jessica4903/)
- [Mingjian Lu](https://www.linkedin.com/in/mingjian-lu-357182102/)
- [Lucas Lu](https://www.linkedin.com/in/lucas-lu-93b867278/)

Division of Pediatric Neurology, Department of Pediatrics, UCLA Mattel Children’s Hospital David Geffen School of Medicine
- [Hiroki Nariai](https://www.uclahealth.org/providers/hiroki-nariai)







