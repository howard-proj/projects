// const sayHello = function() {
//     console.log('Hello');
// }

// const sayHello = () => {
//     console.log('Hello');
// }

// One line function does not need braces
// const sayHello = () => console.log('Hello');

// One line returns
// const sayHello = () => 'Hello';

// Same as above
// const sayHello = function() {
//     return 'Hello';
// }

// Return object
// const sayHello = () => ({msg: 'Hello'});

// Single param does not need parenthesis
// const sayHello = name => console.log(`Hello ${name}`);

// Multiple params need parenthesis
// const sayHello = (firstName, lastName) => console.log(`Hello ${firstName} ${lastName}`);

// sayHello('James', 'Doe');

const users = ['Nathan', 'John', 'Willian'];

// const nameLengths = users.map(function(name) {
//     return name.length;
// });

// Shorter
// const nameLengths = users.map((name) => {
//     return name.length;
// });

// Shortest
const nameLengths = users.map(name => name.length);

console.log(nameLengths);