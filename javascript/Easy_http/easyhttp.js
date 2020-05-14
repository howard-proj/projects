function easyHTTP() {
    this.http = new XMLHttpRequest();
}

// Make an HTTP Get Request
easyHTTP.prototype.get = function(url, callback) {
    this.http.open('GET', url, true);

    // ES6
    // this.http.onload = () => {
    //     if(this.http.status === 200) {
    //         console.log(this.http.responseText);
    //     }
    // }

    // ES5
    let self = this;
    this.http.onload = function() {
        if(self.http.status === 200) {
            callback(null, self.http.responseText);
        } else {
            callback('Error: ' + self.http.status);
        }
    }
    
    this.http.send();
}

// Make an HTTP Post Request
easyHTTP.prototype.post = function(url, data, callback) {
    this.http.open('POST', url, true);
    this.http.setRequestHeader('Content-type', 'application/json');

    let self = this;
    this.http.onload = function() {
        callback(null, self.http.responseText);
    }

    this.http.send(JSON.stringify(data));
}

// Make an HTTP Put Request
easyHTTP.prototype.put = function(url, data, callback) {
    this.http.open('PUT', url, true);
    this.http.setRequestHeader('Content-type', 'application/json');

    let self = this;
    this.http.onload = function() {
        callback(null, self.http.responseText);
    }

    this.http.send(JSON.stringify(data));

}

// Make an HTTP Delete Request
easyHTTP.prototype.delete = function(url, callback) {
    this.http.open('DELETE', url, true);

    let self = this;
    this.http.onload = function() {
        if(self.http.status === 200) {
            callback(null, 'Post Deleted');
        } else {
            callback('Error: ' + self.http.status);
        }
    }

    this.http.send();
}