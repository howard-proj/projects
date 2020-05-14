/**
 * EasyHTTP Library
 * 
 * @version 3.0.0
 * @author Howard J
 * @license MIT
 * 
 */

class EasyHTTP {
    // Make an HTTP Get Request
    async get(url) {
        const response = await fetch(url);
        const resData = await response.json();
        return resData
    } 

    // Make an HTTP Post Request
    async post(url, data) {
        const response = await fetch(url, {
            method: 'POST',
            headers: {
                'Content-type': 'application/json'
            },
            body: JSON.stringify(data)
        });

        const resData = await response.json();
        return resData;

    }

    // Make an HTTP Put request
    async put(url, data) {
        const response = await fetch(url, {
            method: 'PUT',
            headers: {
                'Content-type': 'application/json'
            },
            body: JSON.stringify(data)
        });

        const resData = await response.json();
        return resData;
    }

    // Make an HTTP Delete Request
    async delete(url, data) {
        const response = await fetch(url, {
            method: 'DELETE',
            headers: {
                'Content-type': 'application/json'
            }
        });
        const resData = await 'Resource deleted';
        return resData;
    }
}