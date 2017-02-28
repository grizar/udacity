
// Global variables
var delayData = [];
var dataAllAirports = [];
var createAirportLinesPlaceholder = true;
var currentStory;

var margin = {top: 20, right: 80, bottom: 30, left: 50};
var width;
var height;

	
// D3 objects
var svg;
var tooltip;
var canvas;
var x, xaxis, y, yaxis;
var myline;
var airportLines,path;
var airportCodeToName;

function initStory() {
	
	// Init global variables
	svg = d3.select("svg");
	
	width = svg.attr("width") - margin.left - margin.right;
	height = svg.attr("height") - margin.top - margin.bottom;
	canvas = svg.append("g").attr("transform", "translate(" + margin.left + "," + margin.top + ")");

	// Append average line for all airports and container to receive lines for each airport
	airportLines = canvas.append("g").attr("id","airportlines");
	path = canvas.append("path").attr("id","avgline");
	
	// Change the btn-group style position it below the chart display area
	d3.select(".btn-group").style("width",(width - margin.right - margin.top) + "px")

	// https://bl.ocks.org/mbostock/3884955	

	// Append x axis
	x = d3.scaleLinear().range([0, width]).domain([2004,2015]);
	xaxis = d3.axisBottom(x);
	xaxis.tickFormat(d3.format("0000"));
	canvas.append("g")
		  .attr("class", "axis axis--x")
		  .attr("transform", "translate(0," + height + ")")
		  .call(xaxis);

	// Append y axis
	y = d3.scaleLinear().range([height, 0]).domain([0,0]);
	yAxis = d3.axisLeft(y);
	canvas.append("g")
		  .attr("class", "axis axis--y")
		  .call(yAxis)
		  .append("text")
		  .attr("class","axis--y-label")
		  .attr("transform", "rotate(-90)")
		  .attr("y", 6)
		  .attr("dy", "0.71em")
		  .attr("fill", "#000")
		  .style("font-size","15px")
		  .text("");

	// Define the way a line will be displayed by D3
	myline = d3.line()
		.curve(d3.curveMonotoneX)
		.x(function(d) { return x(d.key); })
		.y(function(d) { return y(d.value); });

	// Add the tooltip. This is the last svg element in order to ensure it will be displayed on top of all other items
	tooltip = svg.append('g');
	tooltip.append('text');

	
	// Compute Airport Code to Airport name lookup
	airportCodeToName = d3.nest()
		.key(function(d) { return d.airportCode})
		.key(function(d) { return d.airportName})
		.rollup(function(d) { return 1 }).entries(delayData);
			
}

function displayData(metric,color,yaxislabel,yaxisformat,top50) {
	// This is the main function of this snippet
	// It takes as parameters the requested metric, the average line color, the y axis label and
	// a flag to mention if we display the TOP 50 aiport data.

	var group = function(item) {
		switch (metric) {
			case 'PercentTotal': 
				return d3.sum(item, function(g) {return g.delay_nas_ct; }) / d3.sum(item, function(g) {return g.flights_ct; });
				break;
			case 'PercentDelay':
				return d3.sum(item, function(g) {return g.delay_nas_ct; }) / d3.sum(item, function(g) {return g.delays_ct; });
				break;
			case 'AvgDur':
				return d3.sum(item, function(g) {return g.delay_nas_dur; }) / d3.sum(item, function(g) {return g.delay_nas_ct; });
				break;
			case '':
				return 0;
				break;
			default:
				// by default, return the total flight number. This case shall never happen
				return d3.sum(item, function(g) {return g.flights_ct; });
		}
	};
	
	// Compute new dataset
	dataAllAirports = d3.nest()
		.key(function(d) { return d.year})
		.rollup(group).entries(delayData);

	// Update y axis. We get the max values of the airport average
	var maxAvg = d3.max(dataAllAirports, function (d) { return d.value });
	var maxY = maxAvg;

	var transition = d3.transition().duration(1500)

	if (top50 == true) {
		// We get the TOP 50 airport data
		dataAirports = d3.nest()
			.key(function(d) { return d.airportCode})
			.key(function(d) { return d.year})
			.rollup(group).entries(delayData.filter(function (d) { return d.airportTop50 == "T";}));
		
		// Pass 1: create the path place holder if needed and find the max value
		dataAirports.forEach(function(d) {
			
			// Let's create the D3 path if it does not relary exists
			if (createAirportLinesPlaceholder) {
				var line = airportLines.append("path").attr("id","line"+d.key).attr("class","airportline").attr("airport",d.key);
				// Append the tooltip display
				line.on('mouseover', function(d) {
					var airportCode = this.getAttribute("airport");
					
					// Get airort name from code
					var airportName = airportCodeToName.filter(function (d) { return d.key == airportCode });
					tooltip.select('text').html(airportName[0].values[0].key);
					tooltip.style('display', 'block');
				});
				line.on('mouseout', function() {
					tooltip.style('display', 'none');
				});
				line.on('mousemove', function(d) {
					// Position tooltip according to real text width
					var textwidth = tooltip.select('text').node().getComputedTextLength()
					var coord = d3.mouse(svg.node());
					
					var tooltipx = coord[0] + 15;
					if ((tooltipx + textwidth) > svg.attr("width")) {
						tooltipx = svg.attr("width") - textwidth;
					}
					var tooltipy = coord[1] + 25;
					tooltip.select('text').attr('y', tooltipy).attr('x', tooltipx);
				});

			}
			
			// Get max value for this aiport. We put a limit on the possible max value, in order to avoid display issues caused by outliers
			var maxAirport = d3.max(d.values, function (d) { return d.value });
			if (maxAirport > maxY) {
				if (maxAirport < 3 * maxAvg) {
					maxY = maxAirport;
				}
			}
		});

		// Set y axis max value
		y.domain([0, 1.2 * maxY]);
		yAxis.tickFormat(d3.format(yaxisformat));
		transition.select(".axis--y").call(yAxis);
		transition.select(".axis--y-label").text(yaxislabel);

		// Display all airports lines
		dataAirports.forEach(function(d) {
			transition.select("#line" + d.key)
				.attr("d", myline(d.values))
				.style("stroke", "lightgrey")
				.style("stroke-width", "2px")
				.style("fill", "none");
		});

		// We are now sure we do not need to create the airport line placeholder
		createAirportLinesPlaceholder = false;
	} else {

		// We do not display the TOP 50 airport. In that case, we only have to set the Y axis max value
		y.domain([0, 1.2 * maxY]);
		yAxis.tickFormat(d3.format(yaxisformat));            
		transition.select(".axis--y").call(yAxis);
		transition.select(".axis--y-label").text(yaxislabel);
	}

	// And finally, display the average line for all airports
	transition.select("#avgline")
		.attr("d", myline(dataAllAirports))
		.attr("stroke", color)
		.attr("stroke-width", "3")
		.attr("fill", "none");        
}


function updateMetrics() {
	// Function called when user press a button
	d3.selectAll(".metric-btn").classed("selected",false); // Unselect all buttons
	var button = d3.select("#" + this.id);
	button.classed("selected",true);
	displayData(button.attr("metric"),button.attr("color"),button.attr("yaxislabel"),button.attr("yaxisformat"),true)
}

function previousstory() {
	story(currentStory - 1);
}
function nextstory() {
	story(currentStory + 1);
}

function story(id) {
	switch (id) {
		case 0:
			// Empty story
			displayData("","black","","",false);
			// And add the story mgt buttons
			d3.select("#next-previous-buttons")
				.append("input")
				.attr("id","btnprevious")
				.attr("type","button")
				.attr("class","button")
				.attr("value", "<< Previous")
				.attr("disabled","")
				.on("click",previousstory);
			d3.select("#next-previous-buttons")
				.append("input")
				.attr("id","btnnext")
				.attr("type","button")
				.attr("class","button")
				.attr("value", "Next >>")
				.attr("disabled","")
				.on("click",nextstory);                
			break;
		case 1:
			// 1st scene: NAS delays % of all flights
			d3.select("#title").html("National Airspace System delays evolution (Expressed as percentage of all flights)");
			d3.select("#storytext1").html("The National Airspace System (NAS) is a major cause of delays representing more than 30% of the delays ...");
			d3.select("#storytext2").html("but from 2004 to 2015, the National Air System caused less and less flight delays ...");
			displayData("PercentTotal","steelblue","Percent",".0%",false);
			d3.select("#btnprevious").attr("disabled",null);
			d3.select("#btnnext").attr("disabled",null);
			break;
		case 2:
			// 2nd scene: NAS delay duration
			d3.select("#title").html("National Air System delays impact (Expressed as the average delay duration)");
			d3.select("#storytext1").html("but at the same time, the average delay duration of National Air System originated delays remains stable.");
			d3.select("#storytext2").html("...");
			displayData("AvgDur","green","Minutes",".0f",false);
			d3.select("#btnprevious").attr("disabled",null);
			d3.select("#btnnext").attr("disabled",null);                
			break;
		case 3:
			d3.select("#title").html("National Air System delays analysis");
			d3.select("#storytext1").html("Grey lines display information for the top 50 airports in term of number of flights. The color line represents the all airports average.");
			d3.select("#storytext2").html("Even if the National Airspace System made efforts to reduce the number of delays, and because the NAS delay average delay duration remain stable, passengers victims of NAS related delays won't notice any change nor improvement.");
			
			// Remove data on the chart
			displayData("","black","","",true);
			
			// Deactivate and hide navigation buttons
			d3.select("#btnprevious").attr("disabled","");
			d3.select("#btnnext").attr("disabled","");     
			d3.select("#next-previous-buttons").style("display","none");

			// Add buttons on the page
			var metrics = [ { label:"Evolution of NAS delays (% of all flights)", metric:"PercentTotal", yaxislabel: "Percent",color:"steelblue", axisformat:".0%"},
							{ label:"Average NAS delays duration (Number of minutes)", metric:"AvgDur", yaxislabel: "Minutes",color:"green", axisformat:".0f"} ];   

			d3.select("#metrics-buttons")
				.selectAll("input")
				.data(metrics)
				.enter()
				.append("input")
				.attr("id",function (d){return "metric-btn-" + d.metric;})
				.attr("metric",function (d){return d.metric;})
				.attr("yaxislabel",function (d){return d.yaxislabel;})
				.attr("yaxisformat",function (d){return d.axisformat;})
				.attr("color",function (d){return d.color;})
				.attr("type","button")
				.attr("class","button metric-btn")
				.attr("value", function (d){return d.label;} )
				.on("click",updateMetrics);           

			d3.select("#PercentTotal").classed("selected",true);
			break;
	}
	currentStory = id;
}

function draw() {
	// Init global variables and launch data loading and display first story
	initStory();
	story(0);
	story(1);
}        

// Row data formatting routine
function formatRow(data) {
	delayData.push({ year: +data.year,
		airportCode: data.airport,
		airportName: data.airport_name,
		airportTop50: data.TOP50,
		flights_ct: +data.arr_flights,
		delays_ct: +data.arr_del15,
		delay_nas_ct: +data.nas_ct,
		delay_nas_dur: +data.nas_delay
	});
}

// Main entry point
function showStory() {
    d3.csv("airline_delay_causes.csv", formatRow, draw);
}

