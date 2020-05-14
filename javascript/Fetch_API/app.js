document.getElementById('button1').addEventListener('click', getText);

document.getElementById('button2').addEventListener('click', getJSON);

document.getElementById('button3').addEventListener('click', getExternal);

// function getText() {
//     fetch('test.txt')
//         .then(function(res) {
//             return (res.text());
//         })
//         .then(function(data) {
//             console.log(data);
//             document.getElementById('output').innerHTML = data;
//         })
//         .catch(function(err) {
//             console.log(err);
//         });
// }

// With arrow function
function getText() {
    fetch('test.txt')
    .then(res => res.text())
    .then(data => {
        console.log(data);
        document.getElementById('output').innerHTML = data;
    })
    .catch(err => console.log(err));
}

// function getJSON() {
//     fetch('posts.json')
//         .then(function(res) {
//             return res.json();
//         })
//         .then(function(data) {
//             console.log(data);
//             let output = '';
//             data.forEach(function(post) {
//                 output += `<li>${post.title}</li>`;
//             });
//             document.getElementById('output').innerHTML = output;
//         })
//         .catch(function(err) {
//             console.log(err);
//         });
// }

function getJSON() {
    fetch('posts.json')
    .then(res => res.json())
    .then(data => {
        console.log(data);
        let output = '';
        data.forEach((post, index) => {
            output += `<li>${post.title} at index: ${index}</li>`;
        });
        document.getElementById('output').innerHTML = output;
    })
    .catch(err => console.log(err));
}

// function getExternal() {
//     fetch('https://api.github.com/users')
//         .then(function(res) {
//             return res.json();
//         })
//         .then(function(data) {
//             console.log(data);
//             let output = '';
//             output += '<ol>';
//             data.forEach(function(user) {
//                 output += `<li>${user.login}</li>`;
//             });
//             output += '</ol>';
//             document.getElementById('output').innerHTML = output;
//         })
//         .catch(function(err) {
//             console.log(err);
//         });
// }

function getExternal() {
    fetch('https://api.github.com/users')
        .then(res => res.json())
        .then(data => {
            console.log(data);
            let output = '<ol>';
            data.forEach((user, index) => {
                output += `<li>${user.login} with index ${index}</li>`;
            });
            output += '</ol>';
            document.getElementById('output').innerHTML = output;
        })
        .catch(err => console.log(err));
}