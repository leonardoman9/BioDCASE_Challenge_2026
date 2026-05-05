Submission
Instructions
Schedule

    14 Feb 2025
    Challenge task descriptions
    01 Apr 2025
    Challenge launch
    01 Jun 2025
    Release of evaluation datasets
    15 Jun 2025
    Challenge deadline
    30 Jun 2025
    Challenge results
    29 Oct 2025
    BioDCASE satellite event

Content

    Introduction
    Submission system
    Submission package
    System outputs
    Technical report

The submission deadline is June 15th 2025 23:59 Anywhere on Earth (AoE)
Introduction

Challenge submission consists of a submission package (one zip package) containing system outputs, system meta information, and technical report (pdf file).

Submission process shortly:

    Participants run their system with an evaluation dataset, and produce the system output in the specified format. Participants are allowed to submit 4 different system outputs per task or subtask.
    Participants create a meta-information file to go along the system output to describe the system used to produce this particular output. Meta information file has a predefined format to help the automatic handling of the challenge submissions. Information provided in the meta file will be later used to produce challenge results. Participants should fill in all meta information and make sure meta information file follows defined formatting.
    Participants describe their system in a technical report in sufficient detail. A template will be provided for the document.
    Participants prepare the submission package (zip-file). The submission package contains system outputs, a maximum of 4 per task, systems meta information, and the technical report.
    Participants submit the submission package and the technical report to BioDCASE2025 Challenge.

Please read carefully the requirements for the files included in the submission package!
Submission system

The CMT submission portal is now closed

Submission guideline:

    Create a CMT user account and login
    Go to the "All Conferences" tab in the system and type BioDCASE to filter the list
    Select "Evaluation and Benchmarking in Automated Bioacoustics"
    Create a new submission

The technical report in the submission package must contain at least the title, authors, and abstract. An updated camera-ready version of the technical report can be submitted separately until 22 June 2025 (AOE).

By submitting to the challenge, participants agree for the system output to be evaluated and to be published together with the results and the technical report on the BioDCASE Challenge website under CC-BY license.
Submission package

The submission package for BioDCASE will follow the same structure as previous DCASE challenges. Participants are instructed to pack their system output(s), system meta information, and technical report into one zip-package. An example submission package is provided:
BioDCASE2025 challenge submission example package (21.1 MB)
(.zip)


Please prepare your submission zip-file as the provided example. Follow the same file structure and fill meta information with a similar structure as the one in *.meta.yaml -files. The zip-file should contain system outputs for all tasks/subtasks, maximum of 4 submissions per task/subtask, separate meta information for each system, and technical report(s) covering all submitted systems.

If you submit similar systems for multiple tasks, you can describe everything in one technical report. If your approaches for different tasks are significantly different, prepare one technical report for each and include it in the corresponding task folder.

More detailed instructions for constructing the package can be found in the following sections. The technical report template is available here.
Submission label

A submission label is used to index all your submissions (systems per tasks). To avoid overlapping labels among all submitted systems, use the following way to form your label:

[Last name of corresponding author]_[Abbreviation of institute of the corresponding author]_task[task number][subtask letter (optional)]_[index number of your submission (1-4)]

For example, the baseline systems would have the following labels:

    Schmid_CPJKU_task1_1
    Nishida_HIT_task2_1
    Politis_TAU_task3_1

A script for checking the content of the submission package will be provided for selected tasks. In that case, please validate your submission package accordingly.
System outputs

Participants must submit the results for the provided evaluation datasets.

    Follow the system output format specified in the task description.

    Tasks are independent. You can participate in a single task or multiple tasks.

    Multiple submissions for the same task are allowed (maximum 4 per task). Use a running index in the submission label, and give more detailed names for the submitted systems in the system meta information files. Please mark carefully the connection between the submitted systems and system parameters description in the technical report (for example by referring to the systems by using the submission label or system name given in the system meta information file).

    Submitted system outputs will be published online on the BioDCASE2025 website later to allow future evaluations.

Technical report

All participants are expected to submit a technical report about the submitted system, to help the BioDCASE community better understand how the algorithm works.

Technical reports are not peer-reviewed. The technical reports will be published on the challenge website together with all other information about the submitted system. For the technical report, it is not necessary to follow closely the scientific publication structure (for example there is no need for extensive literature review). The report should however contain a sufficient description of the system.

Please report the system performance using the provided cross-validation setup or development set, according to the task. For participants taking part in multiple tasks, one technical report covering all tasks is sufficient, if the systems have only small differences. Describe the task-specific parameters in the report.

Participants can also submit a scientific paper to DCASE. In this case, the paper must respect the structure of a scientific publication, and be prepared according to the provided DCASE paper instructions and template.
Template

Reports are in format 4+1 pages. Papers are maximum 5 pages, including all text, figures, and references, with the 5th page containing only references. The templates for technical report are available here:
Latex template (137 KB)
version 1.0 (.zip)


Word template (37 KB)
version 1.0 (.docx)


Sample PDF produced with Latex template (158 KB)
version 1.0 (.pdf)




The Microsoft CMT service was used for managing the peer-reviewing process for this conference. This service was provided for free by Microsoft and they bore all expenses, including costs for Azure cloud services as well as for software development and support.
