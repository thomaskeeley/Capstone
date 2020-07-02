Using Data Science and Remote Sensing to Understand Ship Traffic
with Application to Detecting Forced Labor (Human Trafficking) 
 
# Draft Report
by 
Thomas Keeley & Harry Newton
 
### Abstract
#### Write later

### Problem Statement
By using Machine Learning (ML) on readily available imagery and signals collected by satellites, the presence of ships and their activity can be partially understood.  In this report, we first showcase our work to improve an ML algorithm that operates on imagery based to identify the location of ships.  These results beat the winning submission for a Kaggle competition to do so.  We tuned this algorithm with training data based on assuming the ground truth by the observation of a navigation signal detected at the same location and time.  This navigation signal is the Automatic Identification System (AIS) that most ships emit continuously to comply with international agreements designed to avoid collisions at sea or in port.  The imagery and signals data that we used are described in Table 1.  We describe this first model as our Ship Imagery Model because once tuned, it only requires imagery to identify ships. 

We develop a corresponding Ship Signals Model to summarize the voyages that can be discerned by the AIS data alone, then compare the two models for “scenes” where we have corresponding imagery and signals intelligence at nearly the same time.  We use this Differences Model to find where the location of ships in time and space does not align in the data.

Models like the ones we’ve developed can provide insights into the activity of ships conducting fishing, transport, mining, etc.  There are efforts to use data science for each of these areas.  <Need to refer to lit review to back this up or avoid the statement>.   Of these, we chose to do further work to show how our models could be applied to the fishing industry, particularly to identify practices that could indicate risk factors associated with a ship that may have a crew of forced labor (a type of human trafficking). < Need a much smoother treatment of this once.>
	
### Outline of the Paper
1.	Improved results for Kaggle competition with Ship Imagery Model.  
2.	Ship Signals Model 
3.	Combined Model
4.	Application to Forced Labor detection. 

### Methodology
The research presented in this paper is based on Machine Learning to identify ships based on satellite imagines and Big Data Analysis to operate on satellite signals data.  We employ both of these sensor modalities to improve these techniques iteratively then also used them in combination to discover insights about the ship traffic.  We conclude with a linear model to analyze the data against risk behaviors and present the results using Data Visualization in the form of a Sankey Chart.  This methodology is summarized below. 

### Data Sources 
#### (replace!)
AWS Covid-19 Data Lake	Includes number of cases (new, recovered, deaths) and hospital use & remaining capacity.  Includes work led by Johns Hopkins
Summary of Polaris Hotline calls for past 6.5 years	Conducted by three Ph.D.s on faculty at the Univ of Texas at San Antonio

Polaris Hotline Calls Data	Have appointment next week with officer in charge of this data.  The actual data is highly protected but aggregate measures by location should be available—with caution that areas covered may need to be translated to match Covid-19 data.
Case Mgmt Data from Liberty Shared	Meeting next week with CTO to understand data and infrastructure (Sales Force-based)
Directed Information Graph (DIG) datasets	From work at USC available at Information Sciences Institute webpage.  Includes work for Human Trafficking and a github with scrapped web data 
https://cina.gmu.edu/
Potential other data by CINA at GMU
	

### Improved results for Kaggle competition with Ship Imagery Model 
The application of object detection in high resolution satellite imagery has been explored extensively over recent years. The targets of this type of analysis varies across domains but generally aims at enhancing the ability to process large volumes of satellite imagery data to locate objects of interest. Target objects typically involve vehicles, roads, buildings and vessels. The integration of Computer Vision and Deep Learning frameworks with satellite imagery analysis has yielded innovative results in automatically detecting objects with great accuracy. The object detection application presented in this report focuses on leveraging Geographic Information System (GIS) software tools and Deep Learning models to develop a framework workflow capable of detecting vessels in satellite imagery. 

The innovation of vessel detection in high resolution satellite imagery over recent years has produced very high performing pre-trained models that can be deployed within a user’s environment and applied to a personal use case. Though these high performing models produce benchmark results, they also require a significant amount of computing resources. The application presented in this paper will instead provide the capability to develop and deploy a simpler, more lightweight model that still produces accurate results. The user of this application will be able to either produce their own training data or import from another source, develop and train a Deep Learning model using the Keras framework, deploy the model to run predictions on desired imagery, and finally produce an object detection output with geographic attribution. This type of capability within GIS frameworks has been developed in proprietary software such as ArcGIS. However, the ability to conduct object detection is currently limited in open source GIS software such as QGIS. This application presents the capability to perform this type of analysis on an open source platform with minimal Deep Learning understanding using limited computing resources.

### Ship Signals Model
This will include aligning the resolution of the call center data (at the level that it must be anonymized to protect privacy) and the Covid-19 data.  There will need to be reasoned arguments for which metrics from the Covid-19 data to use because there is not a direct measure of Covid-19 severity.  Initially, it seems like hospital capacity (beds still available) in an area might be a good measure. 

### Combined Model
#### <add later>
	
### Application to Forced Labor Risk Factors
#### <Harry to update…this version is based on original direction which has shifted>
According to Liberty Shared, data on Forced Labor both domestically and internationally is practically non-existent.  This is not for a lack of concern or motivated NGO’s who are willing to collect it, but rather the lack of an appropriate way to collect and curate the data.  A mobile-based data collection/analysis tool could be created to complement the tool that Liberty Shared currently offers for victim case management. This tool could take the form of an add-in to OpenStreetMap or QGIS and would interface with their Victim Case Management System, which is based on Sales Force.  If the idea of this mobile data collection add-in is acceptable to GWU, then we’ll talk with his Chief Technology Officer next week.  We’d start with a list of requirements, including multi-language support and ease-of-use.  Duncan noted that this would be extremely useful for collecting data across the globe on Forced Labor which is currently under-served, and while they have volunteers to collect this data, the data quality and organization does not provide a legitimate data resource.  He has a current effort in Kenya that could help with usability testing. 

Traditional wisdom is that Forced Labor is generally aligned with more traditional commerce, so at the end of the supply chain, the goods are sold using traditional currencies to customers and companies that operate in the open and are willing to stop buying a product or service once a legitimate claim is made about forced labor.  This stands in stark contrast to sex labor which is often has ties to organized crime and crypto currencies.  A reference on this is the recent report by the Financial Threats Council of the Intelligence And National Security Alliance (published May 2020) entitled “Using Intelligence To Combat Trade-Based Money Laundering.”

### Schedule
Task	Duration	Due
1.	Compare Current Call Data to Past 	2 weeks	6/30
Compare Current Call Data to Covid severity	2 weeks	7/14
Propose Data Collection Tool Design	3 weeks	8/4
Prepare Report	2 weeks	8/18
Submit Article	1 week	8/22

### Analysis of Results


### Future Direction
Include any deficits in our work, such as bias to where we can find the data from both sources…
Rubric – <This was for the proposal.  Replace with one for Final Report>
• What problem did you select and why did you select it?	See problem statement above
• What database/dataset will you use?	See table of data sources above.  
• What data science technique will you use to solve the problem?	Machine Learning, Data Analysis using Pandas, Data Visualization (Sankey)
• What framework will you use to implement your work? Why?	Google?
• What reference materials will you use to obtain sufficient background on applying the chosen method to the specific problem that you selected?	Lit Review of data science projects and provided by subject matter experts on HT.
• How will you judge the performance of your work? What metrics will you use?	Against UT-SA results on Polaris. By request for Peer review from CINA at GMU..
• Provide a rough schedule for completing the project.	See schedule table above. 

Save for possible bias of this research for under-reporting.
