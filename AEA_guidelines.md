

Accepted Articles: Manuscript Preparation Guidelines

Please review these guidelines thoroughly before preparing and uploading your manuscript files, supplemental materials, and required documents (see checklist). All materials, with the exception of data and code, are uploaded through ScholarOne Manuscripts.

When preparing manuscripts, authors should follow the Chicago Manual of Style and the AER Style Guide.

Manuscript PDF
The manuscript PDF should include the main text, exhibits, references, and appendices. Manuscripts should not include a title page. Title and byline should be placed at the top of the first page.

Manuscript Native Files
Authors must provide the native files used to create the manuscript PDF.  AEA templates may be used for LaTeX, Scientific Word, and Microsoft Word. 

Data and Code
Authors of conditionally accepted papers that contain empirical work, simulations, or experimental work must provide the data, code, and other details of the computations sufficient to permit replication.  Data and code should be made available and retained in an openly accessible trusted data repository, such as the AEA Data and Code Repository.

As part of the archive, authors must provide a README file listing all included files and documenting the purpose, format, and provenance of each file provided, as well as instructing a user on how replication can be conducted.

See the Data and Code Availability Policy for details.

Data and Code: Related Documents
Authors must provide a signed Data and Code Availability Form and a signed Data and Code Archive Agreement with the final manuscript files.

Author Disclosure Statements
Each (co)author must submit an individual disclosure statement (in PDF format) including the paper title and the author's name at the top. Statements must acknowledge (i) funding sources; (ii) any other potential conflict of interest; and (iii) IRB approval, if applicable. If there is nothing to disclose, an author is still required to supply a disclosure statement formally making that claim.

See the full AEA Disclosure Policy for details and guidance.

Supplemental Appendix
Authors may submit a Supplemental Appendix that contains material such as nonessential extensions, nonessential alternative specifications or robustness checks, and additional discussion that will be of interest to those pursuing work in the area.

The Supplemental Appendix must be provided as a separate PDF and should include a heading that begins with "Supplemental Appendix," followed by the manuscript title and the names of the authors.  Supplemental appendices are posted under "Additional Materials" and are not edited or typeset.

Presentation Slides
Authors may supply optional presentation slides as a separate, clearly labeled PDF.  Presentation slides are posted under "Additional Materials."

Publication Agreements 
When authors fulfill all requirements of conditional acceptance, they will receive a Publication Agreement.  Each (co)author must submit a separate, signed agreement and specify whether implicit or explicit reprint rights are preferred. 
Policy on Revisions of Data and Code Deposits in the AEA Data and Code Repository
 

Policy
No Replacement
Infringement
Identifying That the Deposit Has Been Updated
Minimal Modifications
Review and Approval
Version of Record and Linking
Practical Considerations
Prerequisites: Deposit Ownership
Create a New Version
Uploading Changes
Identifying Changes Made
Publishing Changes

Once an article has been published in an AEA journal, the associated data and code deposit will also have been published in the AEA Data and Code Repository. Both publications are considered permanent.

However, it may become necessary or desirable to update the code and data deposit associated with a published manuscript. Reasons may include:

the code or the data have been updated to more accurately or more easily reproduce the results in the manuscript
data which previously could not be made available is now redistributable
the data availability has changed and the availability statement should be updated
the data are found to contain confidential or copyrighted materials

Policy

No Replacement
All previously published deposits will remain available, except when privacy or copyright infringements have been identified. When updating and publishing a deposit, a new version is created (V2, V3, etc.). All versions connect to each other, for example, if a V2 has been created, there is an indication from the V1 version that it exists, and vice versa.

Infringement
Should a data and code deposit be found to be in infringement of copyright, confidentiality, or other data use agreements, the AEA may be required to “deaccession” or "unpublish" the deposit. The repository will continue to display metadata, including filenames, but the files will no longer be available. Authors are required to bring their deposit back into compliance with the AEA Data and Code Availability Policy as soon as possible.

Identifying That the Deposit Has Been Updated
All data and code deposits are required to have a README. All deposits other than the initial version must also identify the changes made as part of any update.  See "Practical considerations" for details.

Minimal Modifications
Authors should update only those files that need to be added, edited, or removed. Wholesale replacement of the entire archive is not allowed.

Review and Approval
In all cases, the AEA Data Editor will need to review the updated materials provided.

The AEA Data Editor will review the metadata associated with the new version. However, the same level of verification of computational reproducibility afforded to new submissions cannot be provided, except in select cases.

Version of Record and Linking
When the article is published, the approved version of the data and code deposit becomes the version of record (typically "V1"). In the case of a minimal update, the AEA will consider the updated version as the new version of record and will change the link from the article page to point to that version.  For any substantive updates, the link will not be changed.  However, in all cases, all published versions of the data deposit will be visible on the repository page.

Minimal changes include:

clarity of documentation in README or code
­modifications to the deposit that do not affect the computation of the tables and figures in the manuscript
Substantive changes include:

new data or code, even if it improves the overall reproducibility of the package
any modification to data or code that has the potential to change tables and figures

Practical Considerations

Prerequisites: Deposit Ownership
If the materials were deposited with the AEA prior to the announcement of the 2019 Data and Code Availability Policy, the "owner" of the deposit is the AEA Data Editor. In order to update the deposit, authors should request that the AEA Data Editor share the deposit with them.

If the deposit was made after July 2019, one of the authors retains the ability to submit revisions.  The AEA Data Editor can assist in identifying the author who last made changes to the deposit.

Create a New Version
To start the process of revising the deposit, please choose "Change Status" -> "Create New Version" on the top-right corner of the deposit page.

Uploading Changes
When uploading changes, authors should consult the generic AEA Deposit Instructions and supplementary guidance. In particular, we encourage authors to update and enrich any metadata previously not entered, such as geographic scope and time periods covered by the data.

If replacing files, you will first need to delete the original file.
Do not replace anything that does not need replacing (in other words, use surgical replacement rather than bulldozer replacement).
Do not upload ZIP files. All files need to be expanded. You may use the "Import from ZIP" functionality.
The README must be consistent with the new contents of the deposit, and thus usually must be updated as well (see also next section).
Identifying Changes Made
Updates to data and code deposits must exhaustively identify changes made since the Version of Record.  This can be done in one of two ways.

- Create a new file called CHANGES.txt
- Create a new section, near the top of the README, called "Changelog"

Example language:

The data and code in this deposit have been updated after publication of the article.
- V1: Original version
- V2: The code has been simplified, and better instructions provided.  All results are the same.
- V3: Permission was obtained by the data provider to post additional data.  "Master.do" and the instructions in the README have been updated.  Figures 5, 8, and 10 are now reproducible with this archive.

For an example, see the changelog in the README in Tanaka et al (2023) (https://doi.org/10.3886/E148361V2-141003).

Publishing Changes
Please re-submit the deposit to the Data Editor as per supplementary guidance. When publishing a deposit, a new version number is automatically assigned and subsequently displayed.
Policy for Papers Conducting Experiments and Collecting Primary Data
 

Original Instructions
Subject Selection
Software and Scripts
Raw Data
Analysis Programs

For experimental papers and papers that collect primary data via surveys, additional rules apply. We normally expect authors of such articles to supply the following supplementary materials, in addition to the materials required by the standard Data and Code Availability Policy. Any exceptions to this policy should be requested at the time of submission.


Original Instructions
The original instructions for the experiment or survey should be summarized in the submitted manuscript as part of the discussion of the experimental or survey design and should be provided in full as an appendix at the time of submission. Instructions should convey the protocol clearly enough that the design could be replicated by a reasonably skilled experimentalist or survey specialist.


Subject Selection
Information about subject eligibility or selection, such as exclusions based on past participation in experiments, college major, demographic characteristics, etc., should be summarized as part of the discussion of design in the submitted manuscript.


Software and Scripts
Any computer programs, configuration files, or scripts used to run the experiment or develop the survey instrument, e.g., z-Tree code, Qualtrics, and LimeSurvey, shall be provided. These should be summarized as appropriate in the submitted manuscript and deposited in the AEA Data and Code Repository. All requirements noted in the AEA Data and Code Availability Policy for programs apply.

Where appropriate, human-readable versions of the experiment or instrument (PDF of questionnaire) shall also be provided.


Raw Data
The raw data from the experiment or survey should be summarized as appropriate in the submitted manuscript and deposited in the AEA Data and Code Repository, compliant with any confidentiality protections that apply.

We strongly suggest that the deposit separates the raw data and instructions from other replication materials, in order to provide greater visibility to the author's work.


Analysis Programs
All requirements for final and intermediate data files, as well as cleaning and analysis programs, as per the general Data and Code Availability Policy, continue to apply.
Data and Code Availability Policy

Data and Code Availability Policy
Prior to acceptance, authors of papers that contain empirical work, simulations, or experimental work must provide the data, code, and other details of the computations sufficient to permit replication. These materials must be made available and retained in an openly accessible trusted data repository, such as the AEA Data and Code Repository.

These requirements will be adjusted under certain circumstances:

If providing the data publicly is not possible despite the authors' best efforts and due to valid constraints outside the authors' control, authors must (i) commit to preserving data and code for a period of no less than five years following publication of the manuscript, (ii) commit to providing reasonable assistance to requests for clarification and replication, (iii) make the code publicly available, and (iv) publicly document the source of the data, including appropriate contact information. Authors' constraints in making the data publicly available must be noted at the time of submission and may be verified by the AEA data editor prior to acceptance.
In rare cases where any of requirements (i)-(iv) above cannot be met due to valid constraints beyond the authors' control, the AEA data editor and the journal editor may choose to modify the requirement(s), and such modifications will be noted in the article acknowledgements. However, in these cases the source of the data must still be disclosed to the AEA data editor prior to final acceptance of the paper.
The AEA data editor will assess compliance with this policy, including by conducting reproducibility checks and verifying the accuracy of the information provided, and will assist the authors in achieving compliance.


Requirements and Guidelines
The American Economic Association endorses DCAS, the Data and Code Availability Standard v1.0 DCAS 1.0 used by multiple journals in economics, and this data and code availability policy is compatible with DCAS. The specific terms and requirements are described in more detail below. 

Data
Data Availability Statement (DCAS #1)
A Data Availability Statement covering both the source data and any derivative data must be provided either in the README file (see below) or as part of an online appendix. This statement should contain detailed information about data provenance, i.e., how, where, and under what conditions an independent researcher can replicate the steps needed to access the original data, including any limitations and the expected monetary and time cost of data access. This information must be provided even when all data are included as part of the deposit.  

Raw Data (DCAS #2)
Raw data used in the research (primary data collected by the author and secondary data not otherwise available) must be included in the replication package unless the exceptions for non-public data apply (see below) or unless the exact extract of the raw data used in the analysis is published in a trusted repository that satisfies the FAIR data principles (see guidance) and a permanent identifier (e.g., DOI) is provided as part of the Data Availability Statement.

Analysis Data (DCAS #3)
Analysis data should be provided as part of the replication package unless the exceptions for non-public data apply (see below) or unless they can be fully reproduced from accessible data within a reasonable time frame and with reasonable resources.

Non-Public Data
If raw or analysis data cannot be published as part of a replication package or in an openly accessible trusted data repository, the reason(s) must be provided in the Data Availability Statement. Examples include confidential data with identifying information of persons or businesses and data subject to data use agreements or copyrights that prohibit redistribution. It is generally not acceptable that data be provided "upon request" if the request must be approved by the authors themselves. For non-public data, the author should indicate to the AEA Data Editor (in a form provided by the editorial office when requesting final manuscript files) whether a private (not to be published) version of the data can be provided directly to the Data Editor and/or a designated third-party replicator. Please do not upload data to the draft deposit that are not meant for publication.

Formats (DCAS #4)
The data files may be provided in any format compatible with any commonly used statistical package or software. Authors are encouraged to provide data files in open, non-proprietary formats.

Metadata (DCAS #5)
Each variable in the provided datasets should have a meaningful name or description (label), or authors may provide separate codebooks or similar metadata that describe the allowed values and their meaning. It is acceptable to reference publicly available documentation for these items.

Data Citations (DCAS #6)
Please cite all data used in the paper and the approved online appendices as per AEA Reference Style.


Computer Code
A master script is strongly encouraged. When no master script is included, please provide sufficient and precise step-by-step instructions, allowing users to exactly reproduce the generated outputs with the least amount of effort.

When additional packages or libraries are required to run the code, please provide a setup program, containing commands to download and install the necessary packages or libraries.

Code for Data Transformation and Data Cleaning (DCAS #7)
All programs used to generate the analysis data from raw data must be included, even if the raw data cannot be provided.

Code for Analysis (DCAS #8)
Programs that produce computational results such as estimation, simulation, model solution, and visualization must be included. Ideally, these programs reproduce all the computational exhibits in the paper and approved online appendices with minimal human intervention.

Formats (DCAS #9)
The programs may be provided in any format compatible with commonly used statistical packages or software. Should unusual or costly software be required, please notify the AEA Data Editor.

Software Citations
Citation of software packages (e.g., Stata packages, R libraries) is encouraged.


Supporting Materials
Instruments and Experiment Instructions (DCAS #10)
For papers collecting original data through surveys or experiments, the replication materials must include survey instruments or experiment instructions, computer code for experiment or survey collection mechanisms, and original instructions and details on subject selection, unless this information is already provided as part of the paper's appendix. Please see the supplementary Policy for Experimental and Survey Papers.

Ethics Approval (DCAS #11)
If applicable, approval by ethics boards—the Institutional Review Board (IRB) in the United States and equivalent institutions elsewhere—should be demonstrated by including the name of the ethics board and any approval or exemption record number in the title footnote and the author disclosure statement(s). Please see the Disclosure Policy.

Registration of Randomized Controlled Trials (DCAS #12)
It is the policy of the AEA that randomized controlled trials must be registered on the RCT Registry. Please cite all such registrations in the title footnote and elsewhere in the paper as appropriate. For more information, see the RCT Registry Policy.

Documentation (README) (DCAS #13)
Please include a README document in PDF format in the uppermost directory of the replication package. The README file should include the following information:

A Data Availability Statement as described above (or a reference to the appendix containing such information), and statements that the authors had legitimate access to the data for their research and that they have the rights to redistribute the data that is included in the replication package.
A description of the content of the replication package.
An indication of the software and hardware used in the package, including expected running time and specific requirements needed to successfully reproduce the results (software versions, libraries to be installed, etc.). If the requirements and execution time are heterogeneous across significant portions of the package, please indicate specific requirements and running times for each of the different parts.
Instructions on all the steps needed to run the computer code and reproduce all the results.
Information mapping programs to output and how each output relates to the exhibits in the paper and appendices.
Data citations that are not part of the paper itself.
The README must clearly indicate any omission of the required parts of the package due to legal requirements, limitations, or other approved agreements.

While not required, the use of the Social Sciences Data Editors' Template README is strongly encouraged.


Creating a Deposit for the Replication Package
Location (DCAS #14)
The use of the AEA Data and Code Repository is strongly encouraged. Other repositories and archives considered to be "trusted" may be acceptable (see guidance); please contact the AEA Data Editor with any questions. The Data Editor has automatic access to draft deposits created in the AEA Data and Code Repository. If depositing elsewhere, appropriate arrangements should be made to provide the Data Editor with access to the draft deposit.

License (DCAS #15)
Authors retain the copyright to their own data and code and convey any permissions or restrictions imposed on secondary data they include in the replication package. The authors must permit others to use all files in the deposit for the purpose of replication and are encouraged to permit unrestricted access for broader uses. These permissions are recorded in a license. A default license is provided in the AEA Data and Code Repository; other licenses are permissible after review by the Data Editor.

Format
To maximize transparency and accessibility, the data and code must be unzipped within the chosen repository, with few exceptions. Zipped deposits will be returned.

Version of Record
After the data and code deposit is accepted by the AEA Data Editor, it will become the version of record associated with the paper. Corrections and revisions are subject to the Policy on Data and Code Revisions.


Additional Guidance and Documentation
Detailed instructions for preparing and depositing replication packages are provided in the AEA Data Editor's step-by-step guide. Authors are encouraged to reach out to the AEA Data Editor if they believe their particular situation is not covered by the examples and guidance.

For more information, see Frequently Asked Questions.

 

This version (February 2026) supplants all prior data policies.


The AEA encourages institutions to copy and adapt the language in the AEA Data and Code Availability Policy to create their own policies. Use of the AEA name, its logo, and other trademarked items requires permission by the Association.