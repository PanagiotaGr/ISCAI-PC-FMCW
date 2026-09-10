# Αναλυτική έρευνα σχετικής βιβλιογραφίας και θέση του Part B

## Σκοπός

Το παρόν αρχείο εξετάζει αν το **Part B** μπορεί να σταθεί ως αυτόνομη επιστημονική εργασία πάνω στη βάση του υπάρχοντος PC-FMCW ISCAI συστήματος (Part A), ποια κοντινή βιβλιογραφία υπάρχει ήδη και ποιο ακριβώς novelty claim είναι ασφαλές.

Η βασική διάκριση είναι:

- **Part A:** ενοποιημένη PC-FMCW laser-headlamp αρχιτεκτονική για sensing, communication και ADB, με MHT/TBD tracking και κυρίως current-state/reactive λειτουργία.
- **Part B:** causal probabilistic motion forecasting, calibration της predictive uncertainty, propagation της uncertainty σε receiver/angular/beam probabilities και predictive future occupancy, adaptive beam probing και predictive ADB.

Η βασική ερευνητική αλυσίδα του Part B είναι:

```text
PC-FMCW-like sensing observations
        -> measurement uncertainty
        -> probabilistic future-motion prediction
        -> calibrated spatial posterior p(x_{t+tau})
        -> receiver-aware geometry
        -> angular posterior p(theta_{t+tau})
        -> beam probabilities P(B_i)
        -> minimum-cardinality adaptive probing K*
```

Παράλληλα, το ίδιο calibrated future-motion posterior χαρτογραφείται σε future actor occupancy για predictive class-aware ADB.

---

## 1. Βάση: το Part A

### Liu et al., "Phase-coded FMCW Laser Headlamp for Integrated Sensing, Communication, and Illumination"

**IEEE Photonics Technology Letters, accepted 2025. DOI: 10.1109/LPT.2025.3649597**

### Τι κάνει

- Ενοποιεί **sensing, communication και illumination** σε PC-FMCW laser headlamp.
- Ενσωματώνει DPSK data σε FMCW optical waveform.
- Χρησιμοποιεί reflected PC-FMCW signal για sensing/ranging.
- Χρησιμοποιεί **Multidimensional Hough Transform (MHT)** σε track-before-detect λογική για multi-target tracking.
- Περιλαμβάνει ADB που δημιουργεί shadow regions βάσει current target information.
- Δείχνει feasibility με simulation: communication, ranging, tracking και illumination.

### Τι δεν κάνει ως κύριο contribution

- Δεν αναπτύσσει probabilistic future-motion forecasting.
- Δεν κάνει explicit calibration της predictive uncertainty.
- Δεν δημιουργεί receiver-aware angular posterior από future trajectory posterior.
- Δεν προσαρμόζει το beam probing cardinality με βάση posterior probability mass.
- Δεν εξετάζει κοινό calibrated future posterior ως interface για πολλαπλές downstream predictive decisions.

### Συμπέρασμα

Το Part B δεν είναι απλή επανάληψη του Part A. Μπορεί να παρουσιαστεί ως **predictive uncertainty-aware extension** μιας ήδη υπάρχουσας reactive/current-state ISCAI αρχιτεκτονικής.

---

# 2. Κοντινότερη βιβλιογραφία

## 2.1 Wang, Narasimha, Heath — MmWave Beam Prediction with Situational Awareness: A Machine Learning Approach

**IEEE SPAWC 2018. DOI: 10.1109/SPAWC.2018.8445969**

### Τι κάνει

- Χρησιμοποιεί situational awareness σε vehicular περιβάλλον.
- Χρησιμοποιεί θέσεις receiver και surrounding vehicles.
- Μαθαίνει beam information / received power από προηγούμενες παρατηρήσεις.
- Στόχος είναι μικρό beam-training overhead σε υψηλή κινητικότητα.

### Τι σημαίνει για εμάς

Δεν μπορούμε να ισχυριστούμε ότι είναι νέο το:

> location / mobility information -> beam prediction

### Διαφορά Part B

Το Part B διατηρεί explicit probabilistic future state και μεταφέρει τη predictive uncertainty μέσω geometry στο beam domain.

---

## 2.2 Ma et al. — Deep Learning Assisted Calibrated Beam Training for Millimeter-Wave Communication Systems

**IEEE Transactions on Communications, 2021. DOI: 10.1109/TCOMM.2021.3098683**

### Τι κάνει

- CNN prediction από wide-beam measurements.
- LSTM για mobile users.
- Predicted probabilities για narrow-beam refinement.
- Adaptive partial beam training για μείωση overhead.

### Τι σημαίνει για εμάς

Δεν μπορούμε να πούμε ότι είναι νέο:

- probability-assisted beam training,
- adaptive/partial beam training,
- LSTM-assisted mobile beam tracking.

### Διαφορά Part B

Οι beam probabilities μας δεν προκύπτουν απευθείας από received wide-beam patterns. Προκύπτουν από:

```text
calibrated probabilistic motion forecast
    -> receiver geometry
    -> angular uncertainty
    -> discrete beam probabilities
```

Άρα η uncertainty έχει explicit physical interpretation.

---

## 2.3 Xia et al. — Sensing-Enabled Predictive Beamforming Design for RIS-Assisted V2I Systems: A Deep Learning Approach

**IEEE Transactions on Wireless Communications, 2024. DOI: 10.1109/TWC.2023.3327362**

### Τι κάνει

- Χρησιμοποιεί ISAC echo signals σε high-mobility V2I.
- Προβλέπει time-varying CSI.
- Κάνει predictive beamforming για BS και RIS.
- Περιλαμβάνει δύο-stage και end-to-end DL design.
- Στόχος: achievable-rate maximization.

### Τι σημαίνει για εμάς

Δεν είναι νέο το:

> sensing -> prediction -> predictive beamforming

### Διαφορά Part B

Εμείς δεν προβλέπουμε απευθείας CSI ή beamforming vector. Διατηρούμε calibrated physical-state posterior και το μετασχηματίζουμε σε beam-domain probability distribution για adaptive candidate-set sizing.

---

## 2.4 Wang, Wong, Schober — Integrated Sensing and Communications for End-to-End Predictive Beamforming Design in V2I Networks

**IEEE Journal of Selected Topics in Signal Processing, 2024. DOI: 10.1109/JSTSP.2024.3474254**

### Τι κάνει

- Αναγνωρίζει error propagation στο κλασικό sensing-state-estimation -> beamforming pipeline.
- Προτείνει end-to-end DNN που χαρτογραφεί reflected sensing signals κατευθείαν σε future beamformers.
- Παρακάμπτει explicit intermediate state estimation.
- Αξιολογεί achievable sum-rate.

### Γιατί είναι πολύ σημαντικό comparator

Το Part B κάνει συνειδητά το αντίθετο:

- κρατά explicit probabilistic intermediate state,
- το calibrates,
- το χρησιμοποιεί για interpretable risk/coverage-aware downstream control,
- μπορεί να το επαναχρησιμοποιήσει σε διαφορετικό physical function.

Άρα πιθανό research argument είναι ότι το intermediate posterior αξίζει επειδή είναι **interpretable, calibratable και reusable**, όχι επειδή το two-stage pipeline είναι πάντα ανώτερο από end-to-end learning.

---

## 2.5 DeepBeam — A Multi-Agent Deep Reinforcement Learning Framework for Predictive mmWave Beam Management in Dynamic V2X Networks

**IEEE Transactions on Vehicular Technology, 2025. DOI: 10.1109/TVT.2025.3574081**

### Τι κάνει

- Transformer-based trajectory prediction.
- Predictive mmWave beam management.
- Multi-agent deep reinforcement learning.
- Federated learning και coordination.
- Αξιολογεί beam alignment, throughput και latency.

### Τι σημαίνει για εμάς

Δεν μπορούμε να πούμε ότι είναι νέο:

> trajectory prediction -> predictive mmWave beam management

### Διαφορά Part B

Το Part B εστιάζει σε calibrated probabilistic trajectory forecasting και explicit uncertainty propagation αντί για deterministic trajectory prediction / learned policy pipeline.

---

## 2.6 Deng, Shi, Li, Simeone — SCAN-BEST: Efficient Sub-6GHz-Aided Near-field Beam Selection with Formal Reliability Guarantee

**2025, arXiv:2503.13801**

### Τι κάνει

- Χρησιμοποιεί sub-6 GHz channel estimates ως side information.
- CNN προβλέπει probabilities ότι συγκεκριμένα near-field mmWave beams είναι optimal.
- Χρησιμοποιεί **Conformal Risk Control (CRC)**.
- Κατασκευάζει variable-size candidate beam sets.
- Παρέχει formal user-defined target-coverage guarantee υπό τις παραδοχές του.

### Γιατί είναι κρίσιμο

Δεν μπορούμε να ισχυριστούμε ότι είναι νέο:

- probability-based candidate beam sets,
- adaptive candidate-set cardinality,
- target-coverage-oriented beam selection.

Επίσης το δικό μας `q = 0.95` **δεν πρέπει να παρουσιάζεται ως formal finite-sample coverage guarantee**.

### Διαφορά Part B

SCAN-BEST:

```text
sub-6 GHz channel observations
    -> P(optimal beam)
    -> CRC candidate set
```

Part B:

```text
PC-FMCW-like sensing
    -> probabilistic future trajectory
    -> calibrated spatial uncertainty
    -> receiver-aware angular posterior
    -> beam probabilities
    -> minimum-cardinality posterior-mass set
```

Η διαφορά μας βρίσκεται στην **cross-domain uncertainty propagation**, όχι στην ιδέα των variable-size candidate sets από μόνη της.

---

## 2.7 Zhou et al. — Extended Target Adaptive Beamforming for ISAC: A Perspective of Predictive Error Ellipse

**IEEE Transactions on Wireless Communications, 2026. DOI: 10.1109/TWC.2026.3652714**

### Τι κάνει

- Χρησιμοποιεί predictive error ellipses.
- Βασίζεται σε predicted scatterer και communication-receiver positions.
- Προσαρμόζει beamwidth / beamforming σύμφωνα με geometric uncertainty.
- Χρησιμοποιεί minimum enclosing ellipse και adaptive narrowest-beam strategy.
- Αναφέρει achievable-rate gains έναντι conventional beam sweeping.

### Τι σημαίνει για εμάς

Δεν μπορούμε να πούμε ότι είναι νέο:

> predictive geometric uncertainty -> adaptive beamforming

### Διαφορά Part B

Το Part B παράγει explicit **probability distribution over discrete beams** μέσω calibrated trajectory posterior και receiver geometry και επιλέγει minimum-cardinality candidate set που καλύπτει prescribed posterior mass.

---

## 2.8 Cheng, Wu, Zhao — Uncertainty-Calibrated UAV Trajectory Prediction for Beam Management in UAV-Assisted ISAC Scenarios

**Drones, 2026, 10(6), 434. DOI: 10.3390/drones10060434**

### Τι κάνει

- Probabilistic UAV trajectory prediction.
- Explicit uncertainty calibration με split conformal calibration.
- Χρησιμοποιεί calibrated trajectory uncertainty ως risk signal για beam management.
- Adaptive beamwidth / communication decisions βάσει spatial uncertainty.
- Δείχνει βελτίωση σε coverage, outage και misalignment σε high-risk scenarios.

### Γιατί είναι το πιο επικίνδυνο prior work για broad novelty

Αποδεικνύει ότι η ακολουθία:

> trajectory prediction -> calibrated uncertainty -> adaptive beam management

**δεν είναι από μόνη της νέα**.

### Διαφορά Part B

Η δική μας στενότερη contribution chain είναι:

```text
PC-FMCW-like uncertain sensing
    -> calibrated future-motion posterior
    -> receiver-relative transformation
    -> angular posterior
    -> discrete beam posterior
    -> minimum-cardinality adaptive probing
```

Άρα δεν πρέπει να claim-άρουμε novelty στο "calibration + beam management" γενικά.

---

# 3. Predictive ADB και illumination prior art

Adaptive / glare-free headlamp control που χρησιμοποιεί πληροφορία από οχήματα, κάμερες και sensing υπάρχει εδώ και χρόνια. Υπάρχουν επίσης prior approaches και industry work που χρησιμοποιούν trajectory prediction για proactive ADB/glare-free control. Επομένως δεν είναι ασφαλές claim:

> "first predictive ADB"

Η πιο ενδιαφέρουσα θέση του δικού μας ADB branch είναι διαφορετική:

> Χρησιμοποιούμε **το ίδιο calibrated future-motion posterior** που χρησιμοποιείται στο communication branch και το μετατρέπουμε σε future 3D actor occupancy για predictive class-aware ADB.

Το frozen Stage-6 αποτέλεσμα είναι επίσης σημαντικό επειδή είναι αρνητικό ως προς το over-masking criterion:

- vehicle shadow-zone violation: improvement,
- pedestrian visibility: non-inferior,
- cyclist visibility: non-inferior,
- over-masking non-inferiority: **FAIL**,
- reactive over-masking: **0.1886**,
- predictive over-masking: **0.2652**,
- delta: **+0.0766**,
- frozen allowed delta: **+0.0200**.

Αυτό μπορεί να παρουσιαστεί ως scientific finding:

> Η ίδια predictive uncertainty έχει διαφορετικό κόστος ανά physical control domain. Στο communication μπορεί να απορροφηθεί με λίγα επιπλέον probes, ενώ στο illumination η conservative future occupancy αυξάνει το masking.

---

# 4. Συγκριτικός πίνακας

| Χαρακτηριστικό | Part A | Wang 2018 | Ma 2021 | Xia 2024 | Wang/Wong/Schober 2024 | DeepBeam 2025 | SCAN-BEST 2025 | Zhou 2026 | Cheng 2026 | Part B |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Vehicular / mobility-aware beam prediction | Μερικώς | ✓ | ✓ | ✓ | ✓ | ✓ | - | ✓ | ✓ | ✓ |
| Trajectory prediction | - | - | - | - | -/state implicit | ✓ | - | ✓/position prediction | ✓ | ✓ |
| Probabilistic future trajectory | - | - | - | - | - | όχι κύριο | - | geometric uncertainty | ✓ | ✓ |
| Explicit calibration predictive uncertainty | - | - | - | - | - | - | beam-output reliability via CRC | - | ✓ | ✓ |
| Sensing-assisted predictive beamforming | current sensing foundation | - | - | ✓ | ✓ | - | side-information aided | ✓ ISAC | ✓ ISAC | ✓ |
| Spatial -> angular uncertainty propagation | - | - | - | - | - | - | - | related geometric mapping | spatial-risk use | ✓ |
| Explicit beam-domain probability distribution from motion posterior | - | - | predicted beam probabilities from signals | - | - | - | probabilities direct from channel side-info | - | όχι ίδιο discrete chain | ✓ |
| Adaptive candidate-set size / Top-K | - | - | adaptive partial training | - | - | policy-based | ✓ | adaptive width | adaptive width/risk | ✓ |
| Minimum-cardinality posterior-mass rule | - | - | - | - | - | - | related candidate-set concept | - | - | ✓ |
| Formal coverage guarantee | - | - | - | - | - | - | ✓ CRC | - | conformal calibration bounds | **Όχι** |
| ADB / illumination | ✓ reactive | - | - | - | - | - | - | - | - | ✓ predictive |
| Same future posterior reused across comm + illumination | - | - | - | - | - | - | - | - | - | ✓ |
| Frozen negative downstream result retained | - | - | - | - | - | - | - | - | - | ✓ |
| External measured mmWave beam validation | - | - | simulation | simulation | simulation | simulation/realistic evaluation | dataset dependent | simulation | UAV dataset/simulation setup | ✓ DeepSense |

Σημείωση: ο πίνακας είναι novelty-positioning aid και όχι claim απόλυτης παγκόσμιας μοναδικότητας. Πρέπει πάντα να αποφεύγεται wording τύπου "first ever" χωρίς συστηματική και πλήρη bibliographic verification.

---

# 5. Τι ΔΕΝ μπορούμε να claim-άρουμε

Δεν είναι defensible να γράψουμε ότι είμαστε οι πρώτοι που:

1. χρησιμοποιούμε AI για beam management,
2. προβλέπουμε beams από mobility/trajectory information,
3. χρησιμοποιούμε sensing για predictive beamforming,
4. κάνουμε adaptive Top-K / partial beam training,
5. χρησιμοποιούμε uncertainty σε ISAC beamforming,
6. κάνουμε trajectory prediction + uncertainty calibration + beam management,
7. κάνουμε γενικά predictive ADB,
8. εγγυόμαστε 95% frequentist coverage επειδή θέτουμε `q=0.95`,
9. χρησιμοποιούμε real PC-FMCW measurements από WOMD-LiDAR.

---

# 6. Τι μπορούμε να claim-άρουμε με μεγαλύτερη ασφάλεια

Το ισχυρότερο defensible methodological contribution είναι:

> **Αναπτύσσουμε και αξιολογούμε ένα uncertainty-propagated predictive beam-management framework στο οποίο calibrated probabilistic future-motion estimates, προερχόμενα από causal PC-FMCW-like sensing observations, μετασχηματίζονται μέσω receiver-aware geometry σε angular και beam-domain probability distributions. Οι distributions αυτές οδηγούν έναν minimum-cardinality adaptive probing controller που επιλέγει το μικρότερο beam subset το οποίο συγκεντρώνει prescribed posterior probability mass.**

Δεύτερο, system-level contribution:

> **Το ίδιο calibrated future-motion posterior επαναχρησιμοποιείται ως κοινό predictive representation για communication και illumination, αλλά με διαφορετικά downstream mappings και διαφορετικά uncertainty/cost trade-offs.**

Τρίτο, evaluation contribution:

> **Η μεθοδολογία αξιολογείται με classical/learned baselines, calibration analysis, common-support comparisons, frozen sensitivity sweeps, bootstrap statistics και εξωτερικό measured-mmWave validation με DeepSense.**

---

# 7. Μπορεί το Part B να σταθεί μόνο του ως paper;

## Συνοπτική απάντηση: Ναι, μπορεί.

Όμως μπορεί να σταθεί **μόνο αν το paper είναι σωστά εστιασμένο**.

### Αδύναμο framing

Αν παρουσιαστεί ως:

> "Κάνουμε trajectory prediction, uncertainty, beam management και predictive ADB."

τότε το novelty είναι αδύναμο, επειδή σχεδόν όλα αυτά τα components έχουν prior art ξεχωριστά ή σε κοντινούς συνδυασμούς.

### Ισχυρότερο framing

Αν παρουσιαστεί ως:

> **"Επεκτείνουμε μια reactive PC-FMCW automotive ISCAI αρχιτεκτονική σε uncertainty-aware predictive operation και μελετάμε end-to-end propagation calibrated future-motion uncertainty από sensing/mobility space σε receiver-aware angular/beam probabilities και adaptive minimum-cardinality probing. Το ίδιο posterior χρησιμοποιείται παράλληλα για predictive illumination, όπου αποκαλύπτεται διαφορετικό uncertainty–utility trade-off."**

τότε υπάρχει σαφές, διακριτό research question και αρκετό experimental substance για standalone paper.

### Γιατί έχει αρκετό scientific substance

Το Part B περιλαμβάνει:

- real WOMD traffic dynamics,
- model-derived PC-FMCW-like sensing observations,
- measurement uncertainty και predictive uncertainty ως ξεχωριστές έννοιες,
- classical tracking/forecasting baselines,
- deterministic και probabilistic learned predictors,
- uncertainty calibration,
- exact common-support comparisons,
- multiple codebook sizes,
- adaptive posterior-mass probing,
- coverage / uncertainty / codebook sweeps,
- frozen evaluation protocols,
- bootstrap statistical evaluation,
- retained negative ADB result,
- external measured mmWave beam-power validation με DeepSense.

Αυτό είναι επαρκές υλικό για επιστημονική εργασία, εφόσον δεν παρουσιαστούν όλα τα stages ως ισότιμα "novelties".

---

# 8. Προτεινόμενο paper story

## Κύρια ερευνητική ερώτηση

> **Μπορεί calibrated uncertainty για future road-user motion, η οποία προκύπτει από PC-FMCW-like sensing, να μεταφερθεί σε receiver-aware angular/beam probabilities και να χρησιμοποιηθεί για adaptive beam probing με μικρότερο communication overhead χωρίς να θυσιάζεται η επιθυμητή empirical beam coverage;**

## Κύριο methodological contribution

```text
PC-FMCW-like sensing
    -> probabilistic future motion
    -> calibration
    -> receiver-aware angular posterior
    -> beam probability posterior
    -> adaptive minimum-cardinality probing
```

## Δευτερεύον system insight

Το ίδιο posterior τροφοδοτεί predictive ADB, αλλά η uncertainty δεν έχει το ίδιο αποτέλεσμα στα δύο physical functions:

- communication: uncertainty -> λίγα επιπλέον probes -> υψηλή empirical coverage,
- illumination: uncertainty -> μεγαλύτερη πιθανή occupancy region -> περισσότερο masking.

## Προτεινόμενος τίτλος

**Calibrated Uncertainty Propagation for Adaptive Predictive Beam Management in PC-FMCW ISCAI**

Εναλλακτικά, αν θέλουμε να κρατήσουμε πιο έντονα το κοινό communication/illumination story:

**Uncertainty-Aware Predictive PC-FMCW ISCAI: From Future-Motion Posteriors to Communication Beam Management and Adaptive Illumination**

Ο πρώτος τίτλος είναι επιστημονικά πιο focused και πιο ασφαλής ως προς το novelty.

---

# 9. Τελική εκτίμηση

Το Part B **δεν είναι απλώς επέκταση με περισσότερα experiments**. Υπάρχει αυτόνομη ερευνητική ερώτηση, μεθοδολογική αλυσίδα, συγκριτικά baselines, calibration, downstream control και external validation.

Το στοιχείο που πρέπει να θυμάται ένας reviewer δεν είναι ότι "κάναμε πολλά stages". Είναι:

> **Δεν απορρίπτουμε την predictive uncertainty μετά το forecasting. Τη μεταφέρουμε ρητά από το sensing/mobility domain στο receiver/angular/beam domain και τη μετατρέπουμε σε adaptive communication decision. Η επαναχρησιμοποίησή της στο ADB δείχνει επίσης ότι η ίδια uncertainty μπορεί να είναι ωφέλιμη σε ένα physical function και κοστοβόρα σε ένα άλλο.**

Με αυτό το positioning, το Part B μπορεί να σταθεί ως αυτόνομο paper πάνω στο Part A χωρίς να βασίζεται σε μη υπερασπίσιμα "first" claims.
