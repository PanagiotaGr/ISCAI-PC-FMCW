# Frozen Part-A Hough audit for Stage 3

## Frozen source

Part-A repository:

    /home/agni/waymo/part_a_reference/ISCAI_pc_fmcw

Frozen commit:

    44d62e3478e3818d1757b00971890f844cb032f7

Integrity check:

    PASS

## What Part A actually implements

The frozen Part-A Module 2 contains a genuine Hough-based
trajectory reconstruction pipeline.

Observed behavior from the frozen notebook:

1. Synthetic noisy x-y point-cloud detections are generated together
   with random clutter.

2. Each spatial point defines a Hough curve.

3. Voting is accumulated in a Hough parameter space using the
   convention:

       theta approximately in [-90, 90] degrees
       rho may be signed

4. The accumulator is smoothed for robust candidate extraction.

5. Raw Hough peaks are extracted.

6. Candidate peaks are not automatically accepted as tracks.
   A subsequent AND-logic / final track-validation stage is applied.

7. Validated tracks are reconstructed and evaluated using temporal
   frame coverage and trajectory deviation.

8. Part A contains a two-linear-track case with Gaussian noise and
   clutter.

9. Part A also describes a linear/nonlinear case using rolling-window
   stitching.

## Important interpretation

The frozen Part-A implementation is geometric/synthetic Hough tracking
in x-y space.

It is not a measured FMCW Range-Doppler Hough implementation.

Therefore Stage 3 must not claim that the Part-A Hough accumulator
operated directly on measured FMCW data.

## Stage-3 adaptation

Stage 3 preserves the Part-A algorithmic structure:

    raw detections
        -> parameter-space voting
        -> accumulator
        -> peak extraction
        -> suppression
        -> AND-logic temporal validation
        -> reconstructed trajectory

The Stage-3 parameter space is extended to the causal timestamped
trajectory problem:

    [x_anchor, y_anchor, vx, vy]

where x_anchor/y_anchor are defined at the last causal observation time.

This time-aware 4D parameterization is a Stage-3 adaptation required to
distinguish trajectories having different velocities and to support
future trajectory prediction.

The Stage-3 Hough branch:

- consumes raw unlabeled Stage2 detections;
- does not consume estimated GNN associations;
- does not use actor IDs;
- does not use detection keys for temporal matching;
- does not use actor class;
- does not use evaluator truth;
- does not use future observations;
- converts each Stage2 measurement from Ht to the common H0 frame using
  the frozen Stage1 transforms;
- uses Stage2 measurement uncertainty when weighting Hough votes;
- uses radial velocity as an additional consistency term;
- preserves explicit accumulator peaks and final temporal validation.

The final Stage-3 Hough output is a set of causal reconstructed
trajectories suitable for the same downstream trajectory evaluator as
the CV/CA/CTRV/Kalman/IMM baselines.
