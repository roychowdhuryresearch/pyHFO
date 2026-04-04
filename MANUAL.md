# PyHFO 3.0.0 User Manual

This manual is for day-to-day PyHFO use. It is written as an operator guide rather than a developer note. The goal is to answer four practical questions:

1. How do I open data and start working?
2. Which controls matter in the main window?
3. How do I run HFO or spindle workflows safely?
4. What files does PyHFO save and export?

## Document Map

If you are using this manual during real work, the most useful sections are:

- Section 5 if you are new to the application and want a basic screen tour.
- Section 10 if you want the normal HFO workflow.
- Section 11 if you are doing spindle review.
- Section 13 if you want the Quick Detection workflow.
- Section 14 if you are annotating events.
- Section 15 if you are saving or restoring a case.
- Section 16 if you are exporting reports and workbooks.
- Section 17 if you are working with more than one run in the same case.
- Section 20 if something is not working as expected.

## Operator Checklist

For a routine case, PyHFO is safest when you follow this order:

1. Open the recording.
2. Confirm sampling frequency, channel count, and duration.
3. Confirm the biomarker mode.
4. Save a session after the first meaningful processing step.
5. Export only after you know which run is the accepted run.

If you are training someone new, teach them these habits first:

- never annotate before checking the biomarker mode
- never move a `.pybrain` file without its `.pybrain.data` folder
- never send a report HTML file without its `*_report_files` folder
- never assume the active run and accepted run are the same

## 1. What PyHFO Is

PyHFO is a desktop EEG review application for:

- HFO detection and review
- spindle detection and review
- related event classification workflows
- session persistence
- report and workbook export

The current `3.0.0` release centers around one unified workspace instead of the older single-purpose EDF detector layout.

## 2. Supported Inputs And Outputs

### Input recordings

PyHFO can open:

- `EDF` recordings: `.edf`
- BrainVision recordings: `.vhdr`, `.eeg`, `.vmrk`
- `FIF` recordings: `.fif`
- compressed FIF recordings: `.fif.gz`

### Format-specific notes

Use these rules when choosing files:

- For `EDF`, pick the `.edf` file directly.
- For BrainVision, open the `.vhdr` file. PyHFO expects the matching `.eeg` and `.vmrk` files to be present beside it.
- For `FIF`, open the `.fif` file directly.
- For compressed FIF, open the `.fif.gz` file directly.

If a BrainVision case fails to load, check these first:

- the `.vhdr`, `.eeg`, and `.vmrk` files all exist
- the filenames still match each other
- they were not separated into different folders

### Session files

PyHFO can save and load:

- `.pybrain`
- `.npz`

Important detail:

- A `.pybrain` session is not a single file.
- PyHFO writes the main `.pybrain` file plus a companion folder named `<session>.pybrain.data`.
- Keep them together when moving, copying, archiving, or sharing a session.

### Export files

PyHFO can export:

- Excel workbook: `.xlsx`
- event table: `.csv`
- HTML report: `.html`
- waveform snapshot: `.png`
- report asset folder: `*_report_files/`

### Disk layout examples

Typical saved session:

```text
case01.pybrain
case01.pybrain.data/
```

Typical report export:

```text
case01_report.html
case01_report_files/
```

Typical workbook export:

```text
case01_clinical_summary.xlsx
```

## 3. Biomarker Modes

PyHFO supports three biomarker modes from the biomarker selector in the main window.

### HFO

Use `HFO` for the full HFO workflow:

- filtering
- detector configuration
- HFO detection
- artifact / spkHFO / eHFO classification
- annotation
- workbook and report export

### Spindle

Use `Spindle` for spindle workflows based on `YASA`.

This mode supports:

- spindle filter settings
- YASA detector settings
- artifact and spike review support
- annotation
- session save and export

### Spike

Use `Spike` when you want a review-oriented workflow for spike-related events.

Current expectation for `Spike` mode:

- session and review workflows are available
- automated spike detection is not the main supported path in this release

## 4. Installation And Launch

### macOS standalone release

1. Download the release from GitHub Releases.
2. Unzip the downloaded archive if needed.
3. If macOS warns about the app or the DMG, clear quarantine:

```bash
xattr -cr PyHFO-3.0.0-macos-arm64.dmg
```

4. Open the DMG.
5. Drag `PyHFO.app` into `Applications`.
6. Open `PyHFO.app` from `Applications`.

If macOS still blocks the app:

1. Right-click the app.
2. Choose `Open`.
3. Confirm the security prompt.

### Run from source

PyHFO is currently developed around Python `3.9`.

```bash
git clone https://github.com/roychowdhuryresearch/pyHFO.git
cd pyHFO
python3.9 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
python main.py
```

## 5. First Launch: What You See

The main window gives you three immediate entry points:

- `Open File`
- `Load Detection`
- `Quick Detection`

These mean:

- `Open File`: open a new EEG recording from disk.
- `Load Detection`: load an existing saved PyHFO session from `.pybrain` or `.npz`.
- `Quick Detection`: open the smaller one-recording HFO workflow.

After loading a recording, the main workspace exposes:

- waveform controls
- channel controls
- filter controls
- detector controls
- classifier controls
- run statistics
- annotation
- export actions

## 5A. First-Time Startup Scenarios

There are three normal ways to begin.

### Scenario A: new case

Use this when you have a new EEG recording and no prior PyHFO session.

1. Click `Open File`.
2. Load the recording.
3. Confirm the metadata panel is correct.
4. Choose the biomarker mode.
5. Continue with filtering and detection.

### Scenario B: continuing an unfinished case

Use this when you already saved a PyHFO session.

1. Click `Load Detection`.
2. Choose the `.pybrain` file or legacy `.npz`.
3. Wait for waveform initialization to finish.
4. Confirm the correct biomarker mode and run state were restored.

### Scenario C: fast one-off HFO pass

Use this when you only want a single detector run and exports, not the full multi-run workspace.

1. Click `Quick Detection`.
2. Load the recording inside the Quick Detection dialog.
3. Select a detector.
4. Run and export.

## 6. Main Window Anatomy

This section explains the main workspace by screen area.

### 6.1 Top toolbar and startup actions

The main toolbar contains:

- `Open File`
- `Load Detection`
- `Quick Detection`

Typical use:

- Start with `Open File` for a new case.
- Use `Load Detection` to continue prior work.
- Use `Quick Detection` when you want a compact one-pass HFO run instead of the full workspace.

### 6.2 Biomarker selector

At the top of the workspace there is a biomarker selector:

- `HFO`
- `Spindle`
- `Spike`

Use this before configuring detector or classifier settings because available controls depend on the selected biomarker type.

### 6.3 Recording information area

The recording info panel shows:

- file name
- sampling frequency
- number of channels
- recording length

Use this immediately after loading a recording to confirm that the file opened correctly.

### 6.4 Waveform control bar

Above the waveform display, PyHFO provides controls for visual navigation:

- `Number of Channels to Display`
- `Display Time Window`
- `Display Time Window Increment`

Use them to control how much EEG is visible at once.

Practical interpretation:

- fewer displayed channels makes navigation easier
- a smaller time window gives more temporal detail
- the increment controls how far the window advances during navigation

### 6.5 Waveform utility controls

Near the waveform area you will also see:

- `Normalize Vertically`
- `Toggle Filtered`
- `Filter 60 Hz`
- `Bipolar Selection`
- `Choose Channels`
- `Update Plot`
- `N Jobs`

What these do:

- `Normalize Vertically`: rescales traces to a more uniform visual amplitude.
- `Toggle Filtered`: switch the displayed waveform between raw and filtered views.
- `Filter 60 Hz`: apply line-noise suppression in the display workflow.
- `Bipolar Selection`: configure bipolar display pairs.
- `Choose Channels`: restrict the visible channel set.
- `Update Plot`: refresh the current waveform display after control changes.
- `N Jobs`: set the worker count for supported processing steps.

### 6.5A Navigation and review helpers that may appear

Depending on the workspace state and the current build, PyHFO can also expose:

- `Go to time`
- snapshot export buttons
- `Open Review`
- next-pending-event buttons
- run-statistics shortcuts
- accepted-run status badges

These are context-sensitive. If they are disabled, it usually means:

- no recording is loaded yet
- no events exist yet
- no active run exists yet

### 6.6 Overview tab

The `Overview` tab is where most normal operation happens.

It contains:

- `Filter Parameters`
- detector parameter controls
- classifier summary or quick classifier controls
- `Statistics`

### 6.7 Statistics box

The `Statistics` box is the main action area after detection.

It contains:

- `Save As npz`
- `Save As Excel`
- `Annotation`
- `Accept Run` or related run-accept controls in run-management areas

Important historical note:

- The button label still says `Save As npz`.
- In the current main workspace the default save format is actually `.pybrain`, with `.npz` still available as a legacy option in the save dialog.
- Workbook export may auto-mark the active run as accepted if no accepted run exists yet.

### 6.8 Detector tab

The `Detector` tab exposes detector-specific parameter pages. Use this tab when you want to focus on detector settings instead of the compact overview panel.

### 6.9 Classifier tab

The `Classifier` tab contains:

- device setting
- batch size
- default CPU model button
- default GPU model button
- local checkpoint selectors
- Hugging Face model card inputs
- `Use spk-HFO`
- `Use eHFO`
- `Save`

This is the detailed place to configure classification sources.

### 6.10 Log / message panel

The text output area in the main window reports workflow progress and errors. Use it as the first place to look when:

- a run fails
- a detector appears disabled
- loading takes longer than expected

Good operator habit:

- read the log before retrying a failed run
- if a feature is grayed out, the log usually explains whether the issue is missing data, missing dependencies, or missing run state

## 7. Filter Parameters

The standard filter section exposes four important values:

- `Fp`
- `Fs`
- `rp`
- `rs`

In PyHFO these mean:

- `Fp`: pass band frequency
- `Fs`: stop band frequency
- `rp`: pass band ripple
- `rs`: stop band attenuation

Typical defaults in HFO mode are based on:

- `Fp = 80`
- `Fs = 500`

Typical defaults in spindle mode are based on:

- `Fp = 1`
- `Fs = 30`

If you are following a lab protocol, use the protocol values. If not, start from the PyHFO defaults rather than inventing new values.

### Filter sanity check before you run detection

Before you click filter or detect, check:

- `Fp` is below `Fs`
- the values are compatible with the recording sampling frequency
- you are using HFO-oriented defaults in `HFO` mode and spindle-oriented defaults in `Spindle` mode
- you did not accidentally carry spindle settings into HFO mode, or vice versa

## 8. Detector Parameters

PyHFO currently supports `STE`, `MNI`, `HIL`, and `YASA` depending on the biomarker mode.

### 8.1 STE

`STE` exposes the following key fields:

- `sample_freq`
- `pass_band`
- `stop_band`
- `rms_window`
- `min_window`
- `min_gap`
- `epoch_len`
- `min_osc`
- `rms_thres`
- `peak_thres`

Practical reading:

- `rms_window`: RMS calculation window
- `min_window`: minimum event duration
- `min_gap`: minimum separation between events
- `min_osc`: minimum oscillation count
- `rms_thres`: RMS threshold
- `peak_thres`: peak threshold

Use `STE` when you want a standard threshold-based HFO detector with explicit RMS and oscillation controls.

### 8.2 MNI

`MNI` exposes:

- `sample_freq`
- `pass_band`
- `stop_band`
- `epoch_time`
- `epo_CHF`
- `per_CHF`
- `min_win`
- `min_gap`
- `thrd_perc`
- `base_seg`
- `base_shift`
- `base_thrd`
- `base_min`

Practical reading:

- `epoch_time`: analysis epoch duration
- `min_win`: minimum event window
- `min_gap`: minimum event separation
- `thrd_perc`: percentile-style detection threshold
- `base_*`: baseline estimation controls

If you are not already following a validated MNI parameter set, keep the defaults and only change one field at a time.

### Detector parameter change strategy

If you are exploring detector behavior:

1. Duplicate the case logic by running a second detector or a second parameter set.
2. Change only one parameter group at a time.
3. Compare the resulting runs before deciding which run to accept.

Do not change five parameters at once and then try to reason backward from the output.

### 8.3 HIL

`HIL` exposes:

- `sample_freq`
- `pass_band`
- `stop_band`
- `epoch_time`
- `sd_threshold`
- `min_window`

Practical reading:

- `sd_threshold`: standard deviation threshold for the Hilbert-envelope style detector
- `min_window`: minimum accepted event duration

### 8.4 YASA

In spindle mode, `YASA` exposes:

- `sample_freq`
- `freq_sp`
- `freq_broad`
- `duration`
- `min_distance`
- `corr`
- `rel_pow`
- `rms`

Practical reading:

- `freq_sp`: spindle band
- `freq_broad`: broader reference band
- `duration`: allowed spindle duration range
- `min_distance`: separation between spindle events
- `corr`, `rel_pow`, `rms`: YASA threshold terms

## 9. Classifier Configuration

PyHFO supports two ways to define classification models.

### 9.1 Local checkpoint files

Use `Select Model from Your Computer` when you want:

- fully local inference
- fixed model files
- no online dependency at runtime

You can provide:

- artifact model
- spk-HFO model
- eHFO model

### 9.2 Hugging Face model cards

Use `Select Model from Hugging Face Hub` when you want:

- built-in hosted model references
- easier preset-based setup

Default hosted presets are configured around:

- `roychowdhuryresearch/HFO-artifact`
- `roychowdhuryresearch/HFO-spkHFO`
- `roychowdhuryresearch/HFO-eHFO`

### 9.3 Device and batch size

The classifier tab also includes:

- `Device`
- `Batch Size`
- `Use Default CPU Model`
- `Use Default GPU Model`

Use:

- `cpu` when you want the safest default
- `cuda:0` only when CUDA is actually available

If GPU inference is unavailable, stay on CPU.

### 9.4 Optional classifier toggles

The classifier workflow includes:

- `Use spk-HFO`
- `Use eHFO`

Artifact classification is the base requirement when classifier mode is enabled. spkHFO and eHFO are optional add-ons.

### Classifier setup checklist

Before running classifiers, confirm:

- artifact model is configured
- spkHFO is configured if `Use spk-HFO` is enabled
- eHFO is configured if `Use eHFO` is enabled
- the device entry is valid
- batch size is sensible for the machine you are on

If you are unsure:

- use CPU
- use the default hosted model buttons
- keep batch size moderate

## 10. Standard HFO Workflow

This is the recommended full-workspace workflow.

### Step 1. Open the recording

1. Click `Open File`.
2. Select the EEG file.
3. Confirm file name, sampling frequency, channel count, and length in the recording info panel.

### Step 2. Set biomarker mode to HFO

1. Use the biomarker selector.
2. Confirm it is set to `HFO`.

### Step 3. Set waveform visibility

Recommended first adjustments:

- reduce `Number of Channels to Display` if the view is crowded
- choose a manageable `Display Time Window`
- click `Choose Channels` if you only want a subset

### Step 4. Configure filtering

1. Open the filter controls in the `Overview` tab.
2. Review `Fp`, `Fs`, `rp`, and `rs`.
3. Click the filter `OK` button.

### Step 5. Configure the detector

1. Choose the detector you want.
2. Review the detector-specific controls.
3. Start detection.

Recommendation:

- for a fresh case, start with one detector first
- only add comparison runs after you have confirmed the case loaded correctly

What to watch after a first run:

- whether event counts are obviously zero when you expected many events
- whether the event channels make neurophysiologic sense
- whether the waveform overlay looks too dense or too sparse
- whether you accidentally ran the wrong biomarker mode

### Step 6. Review the run

After detection:

- the statistics panel updates
- the event counts and summary fields update
- the `Annotation` button becomes available when events exist

At this point, verify that:

- events were actually found
- the waveform overlays look reasonable
- the recording and event channels make sense

### Step 7. Optional classification

If you need classifier output:

1. Open the classifier tab or classifier controls.
2. Choose local or Hugging Face model sources.
3. Set `Device` and `Batch Size`.
4. Enable `Use spk-HFO` and/or `Use eHFO` if needed.
5. Save classifier settings.
6. Run classification on the active run.

After classification, check:

- whether the run summary changed as expected
- whether artifact-heavy channels now look cleaner in review
- whether the annotation button is still available
- whether the model source you intended is the one that actually ran

### Step 8. Open the annotation window

1. Click `Annotation`.
2. Review events one by one.
3. Save labels as you move through the case.

### Step 9. Accept the run you want to export

If multiple runs exist, choose the one you want to treat as the accepted export run. PyHFO uses the accepted run as the preferred downstream export target.

Do not skip this step in multi-run cases. If you compare `STE`, `MNI`, and `HIL`, the accepted run is the one you are declaring as the preferred export candidate.

### Step 10. Save the session and export outputs

Recommended final sequence:

1. save the session
2. export the workbook
3. export the report
4. export any waveform snapshots you need

## 11. Spindle Workflow

Use this workflow when the case is a spindle review case.

1. Open the recording.
2. Switch biomarker mode to `Spindle`.
3. Confirm the spindle filter settings.
4. Review `YASA` parameters.
5. Run spindle detection.
6. Open annotation if event review is needed.
7. Save the session.
8. Export workbook or report.

Important:

- if `YASA` is unavailable, spindle detection will not run
- the app can still open the case and UI, but the spindle detector path stays disabled

Recommended spindle review pattern:

1. detect
2. inspect the event count
3. review the top channels first
4. annotate a small sample before committing to a full export

## 12. Spike Workflow

Use `Spike` mode mainly for review-oriented work.

What to expect:

- waveform review is available
- session loading and saving are available
- export pipeline is available
- automated spike detection is not the primary release target in `3.0.0`

## 13. Quick Detection Workflow

Quick Detection is a compact HFO-only dialog for single-pass runs.

It is useful when you want:

- one recording
- one detector
- optional classifier
- immediate export

### 13.1 What the Quick Detection dialog contains

Quick Detection includes:

- `Load Recording`
- detector selector
- `N Jobs`
- filter section
- detector-specific sections for `MNI`, `STE`, `HIL`
- classifier section
- export section
- `Run Detection`

### 13.2 Quick Detection workflow

1. Open `Quick Detection`.
2. Click `Load Recording`.
3. Pick one detector from the detector dropdown.
4. Set filter parameters.
5. Adjust detector-specific parameters if needed.
6. Decide whether classifier mode should run.
7. Choose export formats.
8. Click `Run Detection`.

When Quick Detection is the right choice:

- you do not need multi-run comparison
- you do not need the full case workspace
- you want outputs next to the source file quickly

When Quick Detection is the wrong choice:

- you plan to compare more than one detector in the same session
- you expect detailed event curation before export
- you want the richer main-workspace session format

### 13.3 Quick Detection exports

Quick Detection can export:

- `Workbook (.xlsx)`
- `Session (.npz)`

Important difference from the main workspace:

- Quick Detection currently writes session output as `.npz`.
- The main full workspace defaults to `.pybrain` for session saving.

### 13.4 Quick Detection output naming

Quick Detection writes files next to the source recording.

Output names follow this pattern:

```text
<recording_name>_<detector>.xlsx
<recording_name>_<detector>.npz
```

Examples:

```text
case01_ste.xlsx
case01_ste.npz
case01_mni.xlsx
```

If a file already exists, PyHFO appends a numeric suffix instead of overwriting it.

Examples with collisions:

```text
case01_ste.xlsx
case01_ste_2.xlsx
case01_ste_3.xlsx
```

## 14. Annotation Window

The annotation window is the main detailed review tool.

You typically open it after detection or classification results are available.

### 14.1 Main annotation actions

The annotation window includes:

- `Previous`
- `Next`
- `Save and Next`
- `Prev Pending`
- `Next Pending`
- `Clear Label`
- `Prev Match`
- `Next Match`
- `Prediction Scope`
- `Unannotated only`
- frequency range controls
- waveform and FFT panels
- snapshot export

You can think of the annotation window as three jobs combined in one place:

- inspect the waveform
- decide the label
- move efficiently through a queue of events

### 14.2 Annotation labels

In `HFO` mode, the keyboard labels are:

- `1`: Pathological
- `2`: Physiological
- `3`: Artifact

In `Spindle` mode, the keyboard labels are:

- `1`: Real
- `2`: Spike
- `3`: Artifact

### 14.3 Navigation shortcuts

The annotation window supports:

- `Right Arrow` or `D`: next event
- `Left Arrow` or `A`: previous event
- `Enter`: save and move forward
- `Backspace`: clear the current annotation
- `Esc`: clear the FFT ROI

### 14.4 Review helpers

Use:

- `Prev Pending` / `Next Pending` to jump between unreviewed events
- `Prediction Scope` to jump among events matching the selected prediction group
- `Unannotated only` to focus the match navigation on unlabeled events

Recommended annotation strategy for large cases:

1. Use `Next Pending` to move quickly through unreviewed events.
2. If one prediction bucket needs verification, use `Prediction Scope`.
3. Turn on `Unannotated only` when you want to avoid revisiting already-reviewed events.
4. Use `Clear Label` only when you intentionally want to remove a review decision.

### 14.5 Visualization controls

The annotation status bar exposes interaction hints:

- `Shift-drag`: box zoom
- `Alt-drag`: FFT ROI
- mouse wheel: zoom
- drag: pan
- `Esc`: clear FFT ROI

### 14.6 When to use annotation

Open annotation when:

- you need event-by-event decisions
- you want to confirm classifier output
- you want to move from automatic detection to final curated labels

### 14.7 Suggested annotation quality-control pattern

For high-value cases:

1. Review the first 20 to 50 events from the active run.
2. Check whether the label mix is plausible.
3. If the run looks poor, go back and change detector or classifier settings.
4. If the run looks good, continue annotation.
5. Save the session before closing the review window.

## 15. Saving Sessions

### 15.1 Main workspace session save

Use the `Save As npz` button in the statistics area.

Despite the button name, the save dialog now defaults to:

- `.pybrain`

The save dialog still allows:

- `.pybrain`
- legacy `.npz`

### 15.2 What a session stores

PyHFO sessions can preserve:

- biomarker mode
- recording reference
- filter settings
- detector settings
- classifier settings
- active run
- accepted run
- event features
- predictions
- manual annotations

### 15.3 Loading a session

Use `Load Detection` from the toolbar or startup view.

PyHFO restores:

- biomarker mode
- waveform display state
- filter and detector configuration
- classification state when available
- event review readiness

After loading a saved session, always confirm:

- the restored biomarker mode is correct
- the correct recording path was restored
- the waveform view is initialized
- the expected run is active
- the accepted run status still makes sense

### 15.4 Recommended session strategy

For real work, save a session:

- after detection
- after classification
- after a major annotation pass

This makes it easy to resume without recomputing everything.

## 16. Exporting Results

### 16.1 Clinical summary workbook

The main export workbook usually defaults to:

```text
<recording_name>_clinical_summary.xlsx
```

It can include sheets such as:

- `Runs`
- `Channel Ranking`
- `Run Comparison`
- `Decision`
- `Active Run Events`

### 16.2 HTML analysis report

The report export usually defaults to:

```text
<recording_name>_report.html
```

PyHFO also creates a companion asset folder:

```text
<recording_name>_report_files/
```

That folder may contain:

- `clinical_summary.xlsx`
- `events.csv`
- `waveform_snapshot.png`
- `metadata.json`

Do not separate the HTML file from its `*_report_files` folder.

### 16.2A What is inside the report bundle

When data is available, the report bundle may contain:

- `clinical_summary.xlsx`: workbook version of the exported run summary
- `events.csv`: event-level CSV
- `waveform_snapshot.png`: captured waveform image
- `metadata.json`: machine-readable export metadata

This means the report export is not just a webpage. It is a small export package.

### 16.3 Event CSV

The event CSV is generated inside the report asset folder when event-level data exists.

### 16.4 Waveform snapshot

PyHFO can export a PNG image of the current waveform view. This is useful for:

- documentation
- methods supplements
- slide decks
- review notes

## 17. Understanding Active Run Versus Accepted Run

PyHFO distinguishes between:

- the active run
- the accepted run

Practical meaning:

- the active run is the run you are currently viewing or working on
- the accepted run is the run you are choosing as the preferred export target

If you compare multiple detector runs, make sure the accepted run is the one you truly want in the final workbook and report.

## 17A. Multi-Run Comparison Workflow

This is the safest way to compare detectors in the same case.

### Step 1. Create the first run

Run one detector with your baseline settings.

### Step 2. Review the first run briefly

Do not annotate the whole case yet. First check:

- event count
- top channels
- whether the waveform view matches expectations

### Step 3. Create the second run

Run another detector or a modified parameter set.

### Step 4. Open run statistics

Use the run statistics or run management views to compare:

- run counts
- pairwise overlap
- top channels
- active versus accepted run state

### Step 5. Make one run active

Select the run you want to inspect closely.

### Step 6. Accept the preferred run

Use the accept-run action once you know which run should drive export.

### Step 7. Export only after acceptance is correct

This is important because downstream exports prefer the accepted run.

## 17B. Run Statistics and decision logic

When run statistics are available, use them for:

- seeing which runs exist in the case
- checking overlap between runs
- checking which run is active
- checking which run is accepted

If two runs are similar:

- annotate a sample from each before deciding

If one run is clearly better:

- accept it and proceed to export

## 18. Recommended Operator Sequence

If you want one conservative workflow to follow every time, use this:

1. Open the recording.
2. Check recording metadata.
3. Set the correct biomarker mode.
4. Adjust waveform display to a manageable view.
5. Configure filter parameters.
6. Run one detector first.
7. Check whether event output looks plausible.
8. Add classification if needed.
9. Open annotation and review events.
10. Mark the run you want to keep.
11. Save the session.
12. Export workbook and report.

## 19. Practical Tips

### Start simple

For a new case:

- begin with one detector
- keep default classifier settings unless you have a reason to change them
- save a session before trying alternative runs

### Use Go To Time when available

If the `Go to time` field is visible in your current build:

- use it to jump directly to a time point in seconds
- use it when the recording is long and scrolling is inefficient

### Save often during annotation

Annotation work is the most manual part of the workflow. Save after meaningful progress.

### Use snapshots for communication

If you need to discuss a suspicious event with someone else:

- export a waveform snapshot
- save the session
- export the report bundle if a shareable summary is useful

### Keep exported report bundles intact

When sharing a report:

- keep the `.html` file
- keep the matching `*_report_files` folder
- send both together

### Be careful with old filenames

Some labels still use the historical `PyBrain` naming. In practice:

- `.pybrain` is the current session format
- some UI elements still say `npz` or `PyBrain` for historical compatibility

## 20. Troubleshooting

### The app opens but spindle detection is unavailable

Cause:

- `yasa` is missing or not available in the current environment

Fix:

- install the optional dependency in the source environment, or
- use the packaged release that already bundles spindle support

### The session file loads but looks incomplete

Check:

- the `.pybrain` file exists
- the `.pybrain.data` folder exists
- both were kept together

### Quick Detection refuses to start

Common causes:

- no recording loaded
- no detector selected
- no export format selected
- classifier enabled without required model path
- invalid filter or detector field values

### Classifier download fails

Possible causes:

- no internet access
- blocked Hugging Face access
- interrupted first download

Fix:

- retry with network access, or
- switch to local checkpoint files

### The report opens but linked files are missing

Cause:

- the HTML file was moved without its companion asset folder

Fix:

- keep the report HTML and `*_report_files` directory together

### Large cases feel slow

Suggestions:

- reduce the visible channel count
- shorten the displayed time window
- avoid running too many comparison runs at once
- close other memory-heavy applications

### Export produced files but you cannot find them

Check:

- the source recording folder for Quick Detection outputs
- the original recording directory for default workbook and report paths
- the chosen path from the save dialog if you changed it manually

## 20A. File examples

### Example: full workspace save + export

```text
patient001.edf
patient001.pybrain
patient001.pybrain.data/
patient001_clinical_summary.xlsx
patient001_report.html
patient001_report_files/
```

### Example: Quick Detection outputs

```text
patient001.edf
patient001_ste.xlsx
patient001_ste.npz
```

### Example: report bundle contents

```text
patient001_report.html
patient001_report_files/
  clinical_summary.xlsx
  events.csv
  metadata.json
  waveform_snapshot.png
```

## 20B. Training a new operator

If you are teaching PyHFO to someone else, have them practice in this order:

1. open a recording
2. change the biomarker mode
3. change the number of displayed channels
4. run one detector
5. open annotation
6. label ten events
7. save a session
8. export a workbook
9. export a report

That sequence covers almost everything that matters in routine use.

## 21. Version Scope

This manual matches:

- `PyHFO 3.0.0`

This release includes:

- the unified main workspace on the main release line
- validated macOS standalone packaging
- bundled `HFODetector` HIL support
- bundled `YASA` spindle support
