var cols = 25;
var rows = 25;
var grid = new Array(cols);

var openSet = [];
var closedSet = [];
var start;
var end;
// closed set = node that has been visited
// open set = node that requires to be visited

var w, h;
var path = [];

function removeFromArray(arr, elt) {
	for (var i = arr.length - 1; i >= 0; i--) {
		if (arr[i] == elt) {
			arr.splice(i, 1);
		}
	}
}

function heuristic(a, b) {
	// var d = dist(a.x, a.y, b.x, b.y);
	var d = abs(a.x-b.x) + abs(a.y-b.y);
	return d;
}

function Spot(i, j) {
	this.x = i;
	this.y = j;
	this.f = 0;
	this.g = 0;
	this.h = 0;
	this.neighbors = [];
	this.previous = undefined;

	this.show = function(col) {
		fill(col);
		noStroke(0);
		rect(this.x * w, this.y * h, w - 1, h - 1);
	}

	this.addNeighbors = function(grid) {
		var x = this.x;
		var y = this.y;
		if (i < cols - 1) {
			this.neighbors.push(grid[x + 1][y]);
		}
		if (i > 0) {
			this.neighbors.push(grid[x - 1][y]);
		}

		if (j < rows - 1) {
			this.neighbors.push(grid[x][y + 1]);
		}

		if (j > 0) {
			this.neighbors.push(grid[x][y - 1]);
		}	
		
	}
}

function setup() {
	//createCanvas(windowWidth, windowHeight);
	createCanvas(400, 400);
	console.log('A*');

	w = width / cols;
	h = height / rows;

	// Makes 2D array
	for (var i = 0; i < cols; i++) {
		grid[i] = new Array(rows);
	}
	
	for (var i = 0; i < cols; i++) {
		for (var j = 0; j < rows; j++) {
			grid[i][j] = new Spot(i, j);
			//grid[i][j].addNeighbors(grid);
		}
	}

	for (var i = 0; i < cols; i++) {
		for (var j = 0; j < rows; j++) {
			grid[i][j].addNeighbors(grid);
		}
	}

	start = grid[0][0];
	end = grid[cols - 1][rows - 1];


	openSet.push(start);
	console.log(grid);
}

function draw() {
	// background(0, 255, 0);

	
	if (openSet.length > 0) {
		// Keep going
		var lowestIndex = 0;
		for (var i = 0; i < openSet.length; i++) {
			if (openSet[i].f < openSet[lowestIndex].f) {
				lowestIndex = i;
			}
		}
		var current = openSet[lowestIndex];

		if (current === end) {
			noLoop();
			console.log("Finish");

			// // Find the path
			// path = [];
			// var temp = current;
			// path.push(temp);
			// while (temp.previous) {
			// 	path.push(temp.previous);
			// 	temp = temp.previous;
			// }
		}

		// openSet.remove(current);
		removeFromArray(openSet, current);	
		closedSet.push(current);

		var neighbors = current.neighbors;
		for (var i = 0; i < neighbors.length; i++) {
			var neighbor = neighbors[i];

			if (!closedSet.includes(neighbor)) {
				var tempG = current.g + 1;

				if (openSet.includes(neighbor)) {
					if (tempG < neighbor.g) {
						neighbor.g = tempG;
					}
				}
				else {
					neighbor.g = tempG;
					openSet.push(neighbor);
				}

				neighbor.h = heuristic(neighbor, end);
				neighbor.f = neighbor.g + neighbor.h;
				neighbor.previous = current;
				//neighbor.g = current.g + 1;
			}
			//neighbor.g = current.g + 1;
		}
	}
	else {
		// No solution
	}

	background(0);
	for (var i = 0; i < cols; i++) {
		for (var j = 0; j < rows; j++) {
			grid[i][j].show(color(255));
		}
	}
	// debugging purposes: coloring the grid
	for (var i = 0; i < closedSet.length; i++) {
		closedSet[i].show(color(255, 0, 0));
	}

	for (var i = 0; i < openSet.length; i++) {
		openSet[i].show(color(0, 255, 0));
	}

	// Find the path
	path = [];
	var temp = current;
	path.push(temp);
	while (temp.previous) {
		path.push(temp.previous);
		temp = temp.previous;
	}
	
	for (var i = 0; i < path.length; i++) {
		path[i].show(color(0,0,255));
	}
}