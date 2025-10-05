## Paper 103

## Visualizing Clinical Trial Design and Results in 20 Graphical Displays - Achieving 20-20 in 2020

Munish Mehra, PhD (Biostats), M.S (C.S), M.Sc. (Math), Tigermed, USA

## ABSTRACT

With the increasing complexity and amount of data collected in clinical trials there is an explosion in the numbers of Statistical Tables, Figures and Listings (TFLs) generated. Reviewing thousands of pages of TFLs is challenging for Statistical Programmers, Biostatisticians, Clinicians, Medical Writers and others involved in generating, reviewing and summarizing trial design, conduct and results in a clinical study report (CSR). There is a need to present key clinical trial results visually and efficiently. This paper provides an initial approach to do this through 20 graphical displays. The displays include a variety of layout and focus on summarizing the most important information succinctly.

## INTRODUCTION

Clinical trials are conducted for a variety of reasons including establishing whether a new treatment is safe and efficacious. Clinical trials continue to grow in the numbers of subjects enrolled, the complexity of trial design as well as in the amount of data collected. The data collected is summarized in statistical tables, figures and listings across thousands or tens of thousands of pages making it very time consuming and difficult to review and understand the results. This paper proposes a set of 20 standard displays that are recommended to be generated for a typical Phase 2 or 3 trial double blind randomized placebo controlled trial. Event though, the concepts here are for a typical randomized, double blind placebo controlled trial comparing a new treatment to placebo, however the concepts can be also applied to other phases and study designs. The displays below were taken from various publications and online resources as illustrative examples and don't necessarily correspond to the same trial.

There are excellent examples of figures displaying clinical trials data online including from prior PhUSE conferences, PharamSUG, in various publications and at CTSPedia.

The intent of this paper is to share an example of how 20 figures can summarize results from a clinical trial and get input from others involved in designing, programming or reviewed TFLs from a clinical trial to further refine how best to display data from the various domains typically collected as part of a clinical trial.

## TRIAL   DESIGN

Summarizing the key elements of a trial design and results visually include a summary of the trial arms, key timepoints for assessments with what is assed at each timepoint, subject disposition, demographics and other baseline information, efficacy, safety and other key results.

A figure like below displaying the trial treatment arms and the visit structure are often included in most clinical trial protocols. We feel such a figure with adequate detail including adding key interventions or trial data collection provides a quick overview of the trial design.

Figure 1, Trial Design Schema

<!-- image -->

http://tau.amegroups.com/article/view/10481/11773

An additional display such as Figure 2 below is also included in most if not all clinical trial protocols and provides a schedule of assessments including which assessments are performed at each visit. The schedule of assessments often has a lot of footnotes indicating any exceptions or added details of assessments at visits that are relevant.

Figure 2, Schedule of Assessments*

|      | V1 (Screening)   | V2 V1 + 1 Wk (± 3 days)   | V3 V2+1 Wk (± 3 days)   | V4 (Day 1)      | V5 (Wk 8 ± 1 Wk)      | V6 (Wk 12 ± 1 Wk)     | V7 (Wk 16 ± 1 Wk)     |
|------|------------------|---------------------------|-------------------------|-----------------|-----------------------|-----------------------|-----------------------|
|      | Run-in period    | Run-in period             | Run-in period           | Treatment Start | Post-Treatment Visits | Post-Treatment Visits | Post-Treatment Visits |
| IC   | X                |                           |                         |                 |                       |                       |                       |
| MH   | X                |                           |                         |                 |                       |                       |                       |
| DM   | X                |                           |                         |                 |                       |                       |                       |
| AE   | X                | X                         | X                       | X               | X                     | X                     | X                     |
| VS   | X                |                           |                         | X               | X                     | X                     | X                     |
| PRO  | X                |                           |                         | X               | X                     | X                     | X                     |
| etc. |                  |                           |                         |                 |                       |                       |                       |

* The above display is really a Table, however I've included it here as it is the most commonly used method to display the schedule of assessments. For purists on what is a TF or L, often short important listings of individual subject data such as for subjects who died, discontinued or dropped out of a trial along with key data is presented as an individual subject listing in the CSR in 14.3.* as a Table rather than in 16.2.* as a Listing.

## SUBJECT  DISPOSITION

Subject disposition is commonly presented in a CONSORT diagram like below and provides a summary of subjects in each treatment arm, whether they completed the trial or discontinued along with reason for discontinuation. It may add additional details including how many were excluded for key reasons for not meeting INC/EXC criteria, etc. The CONSORT diagram can also include numbers of subjects in the FAS/ITT, mITT, Per Protocol or other populations (analysis sets) analyzed.

Additional details and examples CONSROT diagrams are at: http://www.consort-statement.org/consortstatement/flow-diagram and https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3260171/

The CONSORT diagram is a very valuable figure and should always be included in every clinical trial report.

Figure 3, CONSORT diagram

<!-- image -->

Figure 2: Trial profile

In the open-label phase, all patients receied apomorphine infusion.

Reference: Regina Katzenschlager, et. al, Apomorphine subcutaneous infusion in patients with Parkinson's disease with persistent motor fluctuations (TOLEDO): a multicentre, double-blind, randomised, placebo-controlled trial, Lancet Neurol 2018, http://dx.doi.org/10.1016/S1474-4422(18)30239-4

## SUBJECT  DEMOGRAPHICS   AND   BASELINE   DATA

The primary intent of summarizing demographics and baseline data is to see comparability of treatment groups and the characteristics of the population included in the trial. Demographics and baseline data is either continuous (Age) or categorical (Sex). Continuous data such as Age can also be converted to categorical data such as Age &lt; 65 vs Age ≥ 65. These data can be summarized in bar graphs, box and whiskers plots or Forest plots. Examples of these are below. With the first figure below providing an explanation of the elements of a box and whiskers plot.

<!-- image -->

Different parts of a boxplot

Reference: https://towardsdatascience.com/understanding-boxplots-5e2df7bcbd51

The two figures below indicate alternative ways to present continuous demographics data such as Age by subgroups such as Sex and Race.

Figure 4, Box and Whiskers plot

<!-- image -->

Demographic Profile Layout Panel

<!-- image -->

https://support.sas.com/resources/papers/proceedings09/174-2009.pdf

## EFFICACY   DATA

Many clinical trials are designed to assess efficacy using one or more clinical trial endpoints. The new ICH E9 (R1) guidance ' Addendum on Estimands and Sensitivity Analysis in Clinical Trials to the Guideline on Statistical Principles for Clinical Trials, (November, 2019)' defines the estimand framework that can be used to design a trial with well specified efficacy (or safety) endpoints for regulatory decision making. This framework emphasizes clarity in definition of the estimand and sensitivity analyses to view robustness of results.

## Efficacy analyses may fall into the following broad categories:

1. For continuous endpoints, comparing means (or mean changes from baseline) between treatment arms at a timepoint post treatment. Instead of presenting just raw means and comparing them using a t-test, often least squares means and p-values from a model like Analysis of Variance are used. Sensitivity analyses using different assumptions and subgroup analyses can be displayed visually in Forest plots like below.
2. For categorical variables (or continuous data that has been grouped into categories) one often compares the proportion of subjects between treatment arms who responded to treatment. This can be presented in a variety of ways including bar graphs for categorical variables or by dichotomizing continuous variables using different definitions of 'responders'. One approach to doing this for variables where there may be several clinically meaningful thresholds for defining responders is by presenting the cumulative percentage of responders like below. Forest plots can also be used to display categorical data or dichotomized continuous efficacy or safety data as hazard rations or odds ratios as shown in examples below.
3. In oncology and other therapeutic areas an endpoint of time to event (survival analysis) is often of interest. A forest plot of the hazard ratios or a Kaplan-Meier figure like below are commonly used approaches.

Figure 5, Forest Plot

A Forest plot like below presents the treatment effect for a continues variable (e.g. difference between mean or lsmean change from baseline to last treatment visit between the treatment A and treatment B) along with the 95% Confidence intervals. Results can be presented for the overall population and for clinically meaningful subgroups to see if there are differences. In the figure plot since the vertical line for no treatment effect (vertical line at 0) is within the horizontal lines indicating the 95% CI's, there does not appear to be any treatment effect. The square in the middle of the line displays the mean. Even though there appear to be regional differences this may well be due to chance due to the wide 95% confidence intervals.

<!-- image -->

Reference: https://www.pharmasug.org/proceedings/2019/DV/PharmaSUG-2019-DV-285.pdf

The figure below uses a Forest plot to display hazard ratios for a variable with two possible outcomes (PCI better, Medical therapy better).

| Subgroup          |           | Hazard Ratio                      | 4Yr Cumulative Event Rare   | 4Yr Cumulative Event Rare   | PValue   |
|-------------------|-----------|-----------------------------------|-----------------------------|-----------------------------|----------|
|                   |           |                                   | PCI group                   | Medicaltherpy               |          |
| Overall           |           |                                   | 17.2                        | 15.6                        |          |
|                   |           |                                   |                             |                             | 0,05     |
|                   |           |                                   | 170                         | 13 2                        |          |
| 265 yr            | 632 429)  |                                   | 17,8                        | 213                         |          |
| Sex               |           |                                   |                             |                             |          |
| Male              |           |                                   | 16.8                        |                             |          |
| Female            | 476 /22)  |                                   | 18 3                        |                             |          |
| Nonwhire          |           |                                   | 16.8                        |                             |          |
|                   |           |                                   | 16 7                        | 150                         |          |
| izarion           |           |                                   |                             |                             | 0.81     |
|                   | 963 (44)  |                                   | 18.9                        | 18.6                        |          |
| 27 days           |           |                                   | 15,9                        |                             |          |
|                   |           |                                   |                             |                             | 0,38     |
|                   | 781 436)  |                                   |                             | 16 2                        |          |
| Other             |           |                                   | 15.6                        |                             |          |
| Ejection fraction |           |                                   |                             |                             | 048      |
|                   |           |                                   | 22.6                        | 204                         |          |
|                   | 999 (46)  |                                   | 10.7                        |                             |          |
| Diabetes          |           |                                   |                             |                             | 0,41     |
| Yes               | 446 (21)  |                                   |                             | 23 3                        |          |
| No                | 1720 479) |                                   | 14 4                        |                             |          |
| Killip closs      |           |                                   |                             |                             |          |
|                   |           |                                   | 15.2                        |                             |          |
|                   |           | PCI Better Medical Therapy Botter |                             |                             |          |

https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2222221/pdf/1745-6215-8-36.pdf

Additional examples of forest plots can be found at:

https://pdfs.semanticscholar.org/24b5/14b55498ac2059ce21ec2fae934704c73200.pdf

In the above figure the vertical line at a hazard ratio of 1.0 indicates no treatment effect. In addition, the size of the plotted symbol for point estimates is proportional to the number of events within each subgroup. Forest plots are also used in meta-analyses combining evidence from several related studies.

Reference: Pocock, et. al., How to interpret figures in reports of clinical trials , BMJ. 2008 May 24; 336(7654): 11661169, doi: 10.1136/bmj.39561.548924.94, https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2394578/

## Figure 6, Bar Graph

Figure 6 below is a bar graph and commonly used to present the proportion of subjects in each of the categories indicated on the horizontal axis. Bar graphs may be adjacent as below and with vertical bars or with horizontal bars. Bar graphs may also be stacked to show components of the bar. Bar graphs can also be used to present continuous data such as Change from baseline with a vertical line and horizontal line on top of it to indicate the 95% CI.

Figure 4: Patient Global Impression of Change from baseline to week 12 (full analysis set)

<!-- image -->

Reference: Figure 6 above is from Regina Katzenschlager, et. al, Apomorphine subcutaneous infusion in patients with Parkinson's disease with persistent motor fluctuations (TOLEDO): a multicentre, double-blind, randomised, placebo-controlled trial, Lancet Neurol 2018, http://dx.doi.org/10.1016/S1474-4422(18)30239-4

Stacked and horizontal adjacent bar graphs below are from:

https://support.sas.com/resources/papers/proceedings09/174-2009.pdf

Figures below can effectively display components of a demographic, efficacy or safety variable.

<!-- image -->

The figure below can be used to display incidence rate of a categorical variable by sub-groups such as Sex and by another grouping such as differentiating new cases vs deaths.

Figure 7, Line graphs

<!-- image -->

Line graphs such as below are a common way to present a continuous parameter (such as means, mean change from baseline, or mean % change from baseline) that is collected over time (longitudinal data).

Figure 8, Continuous non-longitudinal data by categories.

<!-- image -->

The figure below presents change from baseline data at a particular time point (Week 12) for the treated and placebo groups based upon the components of the primary endpoint (Hauser diary data indicating a Parkinson patients different states they can be in).

<!-- image -->

Reference: Figures 7 and 8 are from Regina Katzenschlager, et. al, Apomorphine subcutaneous infusion in patients with Parkinson's disease with persistent motor fluctuations (TOLEDO): a multicentre, double-blind, randomised, placebo-controlled trial, Lancet Neurol 2018, http://dx.doi.org/10.1016/S1474-4422(18)30239-4

## Figure 9, Mirror plot

An example of a mirror plot when an endpoint collects data of a subject in different states (conditions or levels of how they feel or function), such as a good state 'On time without troublesome dyskinesia' and a worst state 'Off State', and other in-between states, it is important to see if increases in the good state are also mirrored by decreases in the 'bad state'.

PD motor states at each visit based on home diary results

<!-- image -->

For each variable, data shown are the average from the motor symptom diary for the 2 consecutive days prior to the visit. Error bars are 95% confidence interval for the mean. LOCF last observation carried forward. p-values are APO versus placebo

Reference: Figure 9 is from Regina Katzenschlager, et. al, Apomorphine subcutaneous infusion in patients with Parkinson's disease with persistent motor fluctuations (TOLEDO): a multicentre, double-blind, randomised, placebocontrolled trial, Lancet Neurol 2018, http://dx.doi.org/10.1016/S1474-4422(18)30239-4

## Figure 10, cumulative proportion of responders

Responder analyses are often performed to evaluate the percentage of responders between the different treatment groups. The question often arises whether the threshold chosen was clinically meaningful and what the difference in percentage of responders would be if a different threshold was used. The figure below displays the cumulative percentage of responders for the Placebo and Treated groups for different thresholds. The lower portion of the graph below displays the actual difference in proportion of responders between the two treatment groups along with an indication if it was statistically significant. Overlaying plots like below and using two different Y-axis (% of responders on the left which is for the two lines in the top portion of the plot) and the Treatment effect in hours as indicated on the Y-axis on the right (for the line in the lower portion of the plot) is a common approach to summarizing two sets of data in one plot.

https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3965687/pdf/jsad335.pdf

<!-- image -->

The above information could also be presented as a step down cumulative distribution function like below with 95% CI's included.

Reference: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3125671/pdf/nihms-295566.pdf

<!-- image -->

Figure 11, Kaplan-Meier plot (Survival or Time to event analyses)

In oncology as well as several other trials it is often of interest to visually look at time to an event such as death and compare the occurrence of deaths between treatment groups. When doing a time to event analysis one has to account for censored observations or those for whom only partial information is available such as the subject being alive until the last visit after which they discontinued from the trial and no survival information is available. The publication at https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5045282/pdf/jadp-07-091.pdf provides more on Kaplan-Meir analyses.

In the figure below the proportion of male and female survivors are presented on the Y axis along with the duration in days they were alive along the X axis. In comparing survival between two groups the difference between median survival times or time when 50% of the subjects were still alive are compared. The dotted vertical line below at 0.5 and the corresponding time points on the X-axis where they intersect the two survival curves provide the median survival times. In the figure below it is 270 days for Males and 426 days for Females.

<!-- image -->

Reference for above plot: https://www.quantics.co.uk/blog/introduction-survival-analysis-clinical-trials/

In oncology studies it is important to understand how an outcome variable such as tumor size changes over time for each subject. The figure below displays a spider plot showing relative change in tumor size over time for each subject.

Figure 12, Spider Plot and Spaghetti Plot

Spider Plot, displaying change in tumor size, starting in pre-treatment phase

<!-- image -->

Reference: Nora H and Frankel, Paul H, Graphical Results in Clinical Studies A focus on graphics used in oncology September 2015, Conference: Western Users of SAS Software, San Diego, CA

https://www.researchgate.net/publication/281748374\_Graphical\_Results\_in\_Clinical\_Studies\_A\_focus\_on\_graphics \_used\_in\_oncology

Below is an example of a Spaghetti plot

## Parameter Time Xover

<!-- image -->

https://www.pharmasug.org/proceedings/2013/CC/PharmaSUG-2013-CC27.pdf

Figure 13, Swimmer plots

If additional related outcomes need to be reviewed across time swimmer plots like below can be used.

<!-- image -->

A Durable Responder is a subject who has confirmed response for al least 183 days (6 months) .

Each bar represents one subject in the study.

https://www.pharmasug.org/proceedings/2014/DG/PharmaSUG-2014-DG07.pdf SAS code for Swimmer Plots by Matange, S. (2014, June 22) can also be found at: https://blogs.sas.com/content/graphicallyspeaking/2014/06/22/swimmer-plot/

Additional ways to view key data for each individual subject is by using Waterfall Plots. Subjects responses are ordered (best to worst or worst to best and are displayed as bars with the y axis representing the response and the x axis each subject.

Figure 14, Waterfall plots

## Waterfall Plot in SAS

Fig. 1. Waterfall Plot (Wicklin, R. 2015, April 20)

<!-- image -->

Reference: Wicklin, R. (2015, April 20). Create a waterfall plot in SAS, https://blogs.sas.com/content/iml/2015/04/20/waterfall-plot.html

A waterfall plot can also be used to display difference in responses for subjects in each of the different treatment group by using a different color for each treatment group.

The recent ICH-E(R1) guidance on Estimands and Sensitivity Analyses highlight the importance of performing sensitivity analyses to see how robust the results from primary and key secondary efficacy endpoints are. Forest plots are one way to summarize these. One of the preferred methods of sensitivity analyses is a 'tipping-point' analyses which indicates how changing assumptions in handling missing data can change the conclusions for determining whether a treatment effect is statistically significant or not.

The figure below is an efficient way to present results from a tipping point analysis.

Figure 15, Tipping point sensitivity analysis

<!-- image -->

Mean among all treated (includes imputed missing)

Reference: Torres, Cesar, A Tipping Point Method to Evaluate Sensitivity to Potential Violations in Missing Data Assumptions , ASA Biopharmaceutical Section Regulatory-Industry Statistics Workshop, September 24, 2019

## SAFETY  DATA

Many of the displays used to present demographics or efficacy data can also be used to present safety data, however more efficient ways have been developed for presenting key safety data as assessed by Adverse Events, Laboratory Data, Vital Signs and ECGs. These are presented below.

Figure 16, Dot plot for Adverse Events

The dot plot is significantly underutilized and can convey significant information related to adverse events.

The example below along with R code are available at: https://rdrr.io/cran/HH/man/ae.dotplot.html

## Most Frequent On-Therapy Adverse Events Sorted by Relative Risk

<!-- image -->

Additional examples and details can be found in the book by Jay Herson 'Data and Safety Monitoring Committees in Clinical Trials, where the cover of the book also features a dot plot of AEs.

See: https://www.crcpress.com/Data-and-Safety-Monitoring-Committees-in-Clinical-

Trials/Herson/p/book/9780367261276

The above figure can also be used to present information side by side for related and unrelated events or by severity. In addition presenting dot plots of AEs like above by demographic subgroups (Male vs Female, &lt; vs ≥ 65, etc.), those with a certain medical history or those taking a concomitant medication can be used to explore drug-demographic, drug-disease and drug-drug interactions.

Figure 17, Volcano Plots for AEs.

A volcano plot is a special kind of scatter plot that can display statistical significance such as from a Fisher's Exact test vs the magnitude of change as reflected by the Odds ratio for a large number of tests such as for each adverse event preferred term between the treated and placebo groups. In the figure below the AE's in the upper right quadrant would need further exploration.

<!-- image -->

The link below provides additional examples and SAS code

Reference: https://www.ctspedia.org/do/view/CTSpedia/ClinAEGraph003

Figure 18, Specialized scatter plots to present Shift Tables and Box plots over time to summarize Lab data

The examples below are from 'Use of graphics to understand and communicate clinical research data', Susan Duke, presentation to DIA Clinical Research Community, Nov 2017 and indicate how scatter plots and box plots can be used to summarize Laboratory data. Similar figures can also be used to present vital signs, ECG or other continuous safety data.

## Lab Shift Scatterplots

## Notes on how to use:

- Graphical equivalent of a "shift table"
- Focus on upper left quadrant of each panel
- Reference lines at clinically important levels
- control group last to more readily identify Tx effect (not done in example below) Plot
- May
- Consider displaying on log scale to account for skewness in distribution
- General shift of all points up to the left and
- Distribution of max over several visits (y-axis) has a larger mean than that of a single (baseline) visit (x-axis)

<!-- image -->

## Trellis of Scatterplots for Lab Tests

## Notes on how to use:

- be useful with any lab data to pick up possible associations May
- Allows identification of simultaneous elevations
- Similar graphical features to Trellis plot
- Will normalize lab results by either Upper Limit of Normal Range or Upper Limit of PCI Range and graph each lab test against the others in a matrix.

## Lab Boxplots over Time

## Notes on how to use:

- Standard boxplots (Tukey's schematic diagrams)
- Colour and symbol to differentiate treatments
- Time axis on continuous scale (not nominal)
- Numbers of patients for each time point displayed under x-axis
- Graphical margin shows distribution of maximum change for each patient
- Number of extreme values (22 ULN) noted above the plot

<!-- image -->

<!-- image -->

## Side-by-side Boxplots

Notes on how to use:

- The main interest is in the outliers the central mass of the observations can be relegated to a simple box
- Alternative display to scatterplots
- In comparison to scatterplots , includes additional information about median and distribution across all subjects
- Similar graph could be used to examine muscle by profiling creatinine phosphokinase (CPK); ASAT, lactate dehydrogenase (LDH) injury

<!-- image -->

A bubble plot is a special type of scatter plot where the size of the dots are represented as bubbles proportional to a response variable. Bubble plots are not as useful since our eyes have difficulty comparing differences in areas. See example below.

Ref: Mukul K Mittal, Efficacy endpoint visualization of investigational products for cancer patients using R Shiny, Sarah Cannon Development Innovations, PhUSE US Connect, Baltimore, MD. February 2019

<!-- image -->

## Figure 19, ECG Plot

The ICH E14 guidance provides both absolute thresholds and thresholds of change from baseline that would be considered as ECG abnormalities. The figure below displays baseline QTc on the X-axis for each subject and QTc Changes along the Y-axis. The dotted lines in the upper right hand quadrant correspond to absolute thresholds reached for a given baseline on the X-axis and a change as reflected on the Y-axis. The same concept can also be used to evaluate vital signs abnormalities.

<!-- image -->

The above image and code are available at: https://www.ctspedia.org/do/view/CTSpedia/ClinECGGraph000

Another special type of scatter plot can be used to evaluate liver function abnormalities using Hy's law is displayed in the figure below.

## Plots to Identify Potential Hy's Law Cases

Ref: Alan M Shapiro, MD, PhD, FAAP, Safety Review: Approach &amp; Tools, PhUSE CSS, June 10, 2019

<!-- image -->

<!-- image -->

The above presentation also shares a safety analysis toolbox in R Shiny used by FDA (See below)

## Analysis Toolbox R Shiny Tools

<!-- image -->

<!-- image -->

With the growth in wearable devices there is large volume of data collected over time continuously. One way to represent this for long durations is using a Spiral plot like below where time is represented in spirals. In below plot each circle is a 24 hr. day and different circles correspond to different days.

## Time-Series Plots

Spiral Plot

<!-- image -->

Hour

Schramek, Dan A Future 'View' of Digital Data from Wearable Devices, GSK, 2019 PhUSE US Connect, Baltimore, MD. February, 2019

Figure 20, Patient Profiles

Patient Profiles can be used to present individual subject disposition, demographics, safety and efficacy data like below:

<!-- image -->

Reference: www.patientprofiles.com downloaded 4 March, 2009, Can be accessed on WayBackMachine at: https://web.archive.org/web/20091104034528/http://www.patientprofiles.com/examples.html

<!-- image -->

Additional ways to display patient profiles data using swimmer plots are below:

## Swimlane Charts Clearly Relate Events Over Time

Patient Profile Connect Exposure; Labs; Conmeds &amp; AEs Temporally

<!-- image -->

<!-- image -->

- Swimlane charts allowing onset and duration of events to be compared
- Potential correlations between exposure, conmeds, and AEs can be rapidly identified
- Simple charts allow rapid monitoring as the progresses study

Data Visualization PhUSE CONNECT 2019 Baltimore

The graphs rendered herein are not referenced to any specific GSK study or asset, contain no personally identifiable information, and are shown for demonstrative purposes only -

## Swimlanes Aggregate Large Amounts of Information

Duration on Treatment by Treatment Status

<!-- image -->

<!-- image -->

- Swimlanes of treatment duration by status provide excellent overviews of individual patient milestones and events
- Utility is compromised when multiple filters need setting to access desired view
- Multiple pages provide functionality but add considerable overhead
- Creation of FilterlView Presets (e.g. Filters for Presentation; above) with Python provides new controls for rapid navigation and view customization
- Controls allow page

Data Visualization PhUSE CONNECT 2019 Baltimore

31

The graphs rendered herein are not referenced to any specific GSK study or asset; contain no personally identifiable information; and are shown for demonstrative purposes only.

Szewczyk, Jason W. , Delivery of Data Visualizations within GlaxoSmithKline Clinical Development and Partners, PhUSE CONNECT US, Baltimore, MD, 24 -27 February 2019

In addition to the various figures presented above and the related references and links at CTSPedia and PhUSE there are excellent presentations to see how best to use figures to communicate clinical trial results. Some of these are indicated below:

'Seeing Is Believing! Good Graphic Design Principles for Medical Research', Susan Duke et. al at https://www.ctspedia.org/wiki/pub/CTSpedia/GraphicsPresentationArchive/DIA2014\_Susan\_Duke\_Graphics.pdf

'Graphical and Quantitative Literacy: Empowerment Begins with Naming and Describing What We Do', Joint Statistical Meetings, Vancouver, BC, July 29, 2018, Susan Duke, Mathematical Statistician, Office of Biostatistics, CDER, FDA.

'A Picture is Worth a Thousand Tables, Graphics in Life Sciences', Editors Andreas Krause, Michael O'Connell, Springer, 2012

Note: FDA reviews such as at https://www.accessdata.fda.gov/drugsatfda\_docs/nda/2015/203312Orig1s000MedR.pdf also provide a summary of medical, statistical and other reviews where one can find useful information on statistical methodology used as well as Tables and Figures FDA statisticians used to support their summary reviews.

## CONCLUSION

This paper provides a way to summarize visually in a concise way the key elements of a trial design and the safety and efficacy results in 20 displays. Reviewing these displays provides a quick overview for anyone wanting to understand the design and results from a given clinical trial. Medical Writers, Clinicians and Biostatisticians typically review most of the Tables that are including in-text in the CSR or in post-text Appendices. Viewing a clinical trial design and results in 20 displays such as presented here allows Statistical Programmers, Data Managers, Clinical Monitors, Clinical Operations personnel, Project Management, Quality Assurance, Regulatory personnel and others to very quickly review the culmination of all their work and see if from their perspective how the final results of a clinical trial are summarized. Other figures may be appropriate in different settings that we would be happy to learn about and include in future refinements of this paper.

For most of the above 20 Figures, supporting Tables and Listings can provide the exact numbers and other necessary statistics.

The above figures could also be implemented as interactive graphs with drill down capability, but may increase the complexity and ability to succinctly view results from a given trial in addition to reviewers having to learn a new application.

Our future efforts will include applying the concepts here to a single trial and getting feedback from all those involved in a trial.

## CONTACT  INFORMATION

Your comments and questions are valued and encouraged to find the best way to summarize a clinical trial in 20 Figures.  Contact the author at:

Author Name: Munish Mehra

Company: Tigermed

Address: 12500 Copen Meadow Ct.,

City / Postcode: North Potomac, 20878

Work Phone: +1-240-477-3700

Email: munish.mehra@tigermed.net

Web: www.tigermedgrp.com

Brand and product names are trademarks of their respective companies.