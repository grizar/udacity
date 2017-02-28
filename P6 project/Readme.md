#US National Airspace System delays (NAS) analysis

##Summary
This analysis will show that US National Airspace System related delays tends to decrease other the time.
Nevertheless, the impact of the National Airspace System delay for the passenger remains the same over the years.

Because a single passenger will be impacted by few NAS related delays and the delay duration remain the roughly the same, he won't be able to notice the efforts made by the National Airspace System to improve.

##Initial design decision

To support my findings, I wanted to show 3 different metrics:
* The proportion of NAS related delays vs the total number of flights
* The proportion of NAS related delays within the delayed flights
* The impact of the NAS related delays expressed in average delay duration.

According to finding, the initial design decision was to use a simple trend line to show the different metrics over the years.
We show the 3 different metrics for all airports in a sequential manner. A different color code is used for each metric. 
We let then the user select the metrics he wants to analyse. At this point, I added lines for the 50 most frequented airports.

Finally, the 2nd metric has been removed because too confusing.

##Received feedback

###Feedbacks received from Allan H on v1:
Comment: Button usage not clear.
Answer : Button layout modified.

Comment: Put a % sign of the Y axis when needed.
Answer : Y axis format set.
 
Comment: User shall be able to move from one scene to another by himself.
Comment: Navigation trough previous/next buttons added.

Comment: Do not understand the correlation between the 3 charts.
Answer : They objective was not to provide an answer of this strange behaviour. Opening sentence added at end of story 3.
Comment: Button wording to be reviewed.
Answer : Button wording review

Comment: Animation are appealing. 

###Feedbacks received from Gregoire R on v2:

Comment: Slide 2 is confusing. Shall be either removed or put in 1st position and reworked.
Answer : Reordering the slides.

Comment: Title shall change according to slide
Answer : Size set.

### Feedbacks received from Guy D on v2:

Comment: Second slide is confusing.
Answer : remove 2nd slide and associated KPI.

Comment: Display airport when mouse is on an airport line. It can be interesting to know which airport is concerned.
Answer : Tooltip added
