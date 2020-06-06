var cols = 50;
var rows = 50;
var grid = new Array(cols);

var openSet = [];
var closedSet = [];
var start;
var end;
// closed set = node that has been visited
// open set = node that requires to be visited

var w, h;
var path = [];
// var noSolution = false;

function removeFromArray(arr, elt) {
	for (var i = arr.length - 1; i >= 0; i--) {
		if (arr[i] == elt) {
			arr.splice(i, 1);
		}
	}
}

function heuristic(a, b) {
	var d = dist(a.x, a.y, b.x, b.y);
	// var d = abs(a.x-b.x) + abs(a.y-b.y);
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
	this.wall = false;

	if (random(1) < 0.3) {
		this.wall = true;
	}

	this.show = function(col) {
		// color the spot 1) red in closed set, 2) blue as a path, 3) green as an open set, 4) obstacle as black
		fill(col);
		// If a wall, make it black
		if (this.wall) {
			fill(0);
			// noStroke();
			// ellipse(this.x * w + w/2, this.y * h + h/2, w/2, h/2);
		}
		noStroke();
		rect(this.x * w, this.y * h, w - 1, h - 1);
		// ellipse(this.x * w + w/2, this.y * h + h/2, w/2, h/2);
	}

	this.addNeighbors = function(grid) {
		var x = this.x;
		var y = this.y;

		// Adjacents
		if (x < cols - 1) {
			this.neighbors.push(grid[x + 1][y]);// left
		}
		if (x > 0) {
			this.neighbors.push(grid[x - 1][y]);// right
		}

		if (y < rows - 1) {
			this.neighbors.push(grid[x][y + 1]);// bottom
		}

		if (y > 0) {
			this.neighbors.push(grid[x][y - 1]);// top
		}

		// Diagonals
		if (x > 0 && y > 0) {
			this.neighbors.push(grid[x - 1][y - 1]); //top left
		}

		if (x > cols-1 && y > 0) {
			this.neighbors.push(grid[x + 1][y - 1]); // bottom right
		}

		if (x > 0 && y < rows - 1) {
			this.neighbors.push(grid[x - 1][y + 1]); // bottom left
		}

		if (x < cols - 1 && y < rows - 1) {
			this.neighbors.push(grid[x + 1][y + 1]); // top right
		}

		
	}
}

function setup() {
	//createCanvas(400, 400);
	createCanvas(windowWidth, windowHeight);
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
	start.wall = false;
	end.wall = false;


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

			if (!closedSet.includes(neighbor) && !neighbor.wall) {
				var tempG = current.g + 1;

				var newPath = false;
				if (openSet.includes(neighbor)) {
					if (tempG < neighbor.g) {
						neighbor.g = tempG;
						newPath = true;
					}
				}
				else {
					neighbor.g = tempG;
					newPath = true;
					openSet.push(neighbor);
				}

				if (newPath) {
					neighbor.h = heuristic(neighbor, end);
					neighbor.f = neighbor.g + neighbor.h;
					neighbor.previous = current;
				}
				
				//neighbor.g = current.g + 1;
			}
			//neighbor.g = current.g + 1;
		}
	}
	else {
		// No solution
		console.log('No solution');
		// noSolution = true;
		noLoop();
		return;
		
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
	// if (!noSolution) {
	path = [];
	var temp = current;
	path.push(temp);
	while (temp.previous) {
		path.push(temp.previous);
		temp = temp.previous;
	}
	// }
	
	
	for (var i = 0; i < path.length; i++) {
		path[i].show(color(0,0,255));
	}

	// noFill();
	// stroke(255);
	// // strokeWeight(w/2);
	// beginShape();
	// for (var i = 0; i < path.length; i++) {
	// 	vertex(path[i].x*w + w/2, path[i].y*h + h/2);
	// }
	// endShape();
}