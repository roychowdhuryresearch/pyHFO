# PyHFO
<img src="src/ui/images/icon1.png" alt="pyHFO logo" style="width:30%;">

[ProjectPage](https://roychowdhuryresearch.github.io/PyHFO_Project_Page/) |
[Download](https://github.com/roychowdhuryresearch/pyHFO/releases) | 
[Manual](https://docs.google.com/document/d/1KzQpfuPFDk2lr9V3TgSkmc21jISB54ZOxo3pIYKQsp0/edit?usp=sharing) |

PyHFO is a multi-window desktop application providing an integrated and user-friendly platform that includes time-efficient HFO detection algorithms such as short-term energy (STE) and Montreal Neurological Institute and Hospital (MNI) detectors and deep learning models for artifact and HFO with spike classification.

## Bibtex
If you find our project is useful in your research, please cite:

```
Zhang, Y., Liu, L., Ding, Y., Chen, X., Monsoor, T., Daida, A., Oana, S., Hussain, S. A., Sankar, R., Fallah, A., Santana-Gomez, C., Engel, J., Staba, R. J., Speier, W., Zhang, J., Nariai, H., & Roychowdhury, V. (2024). PyHFO: lightweight deep learning-powered end-to-end high-frequency oscillations analysis application. Journal of neural engineering, 10.1088/1741-2552/ad4916. Advance online publication. https://doi.org/10.1088/1741-2552/ad4916
```

## Related Projects

* [HFODetector](https://github.com/roychowdhuryresearch/HFO_Detector) - A Python toolbox for very fast HFO detection.

* [HFO-Classification](https://github.com/roychowdhuryresearch/HFO-Classification) - Many HFO classification projects powered by deep learning.

* [EEG-Viz](https://github.com/jebbica/EEG-Viz) - A Python toolbox for EEG visualization.


## Installation

You can download the latest version of PyHFO from the [releases](https://github.com/roychowdhuryresearch/pyHFO/releases) page.

If you choose to use the **macOS version** of the standalone distributable application, please follow these additional steps:

1. **Download and unzip** the `.zip` file.
2. You will get a file named `pyHFO.dmg`.
3. Navigate to the directory containing the `pyHFO.dmg` file.
4. Open the terminal and run the following command to remove the quarantine attribute:

```
xattr -cr pyHFO.dmg
```

You can also install it from the source code:

```
git clone https://github.com/roychowdhuryresearch/pyHFO.git 
cd pyHFO
conda create -n pyhfo python=3.9
conda activate pyhfo
pip install -r requirements.txt
python main.py
```
If you encounter failure in installing pyqt, please do pip install pyqt and remove pyqt related lines in requirements.txt

## Usage

The overview of the PyHFO is shown below:
![Alt text](img/overview1.png)


The manual is available [here](https://docs.google.com/document/d/1KzQpfuPFDk2lr9V3TgSkmc21jISB54ZOxo3pIYKQsp0/edit?usp=sharing).

## License

This project is licensed under the UCLA Academic License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

### Contributors:
This project is under supervsion of Prof. [Vwani Roychowdhury](https://www.ee.ucla.edu/vwani-p-roychowdhury/).

Department of Electrical and Computer Engineering, University of California, Los Angeles
- [Yipeng Zhang](https://zyp5511.github.io/)
- [Lawrence Liu](https://www.linkedin.com/in/lawrence-liu-0a01391a7/)
- [Yuanyi Ding](https://www.linkedin.com/in/yuanyi-ding-4a981a132/)
- [Xin Chen](https://www.linkedin.com/in/xin-chen-980521/)
- [Jessica Lin](https://www.linkedin.com/in/jessica4903/)

Division of Pediatric Neurology, Department of Pediatrics, UCLA Mattel Children’s Hospital David Geffen School of Medicine
- [Hiroki Nariai](https://www.uclahealth.org/providers/hiroki-nariai)







