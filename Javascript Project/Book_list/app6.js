class Book {
    constructor(title, author, isbn) {
        this.title = title;
        this.author = author;
        this.isbn = isbn;
    }

}

class UI {
    addBookToList(book) {
        const list = document.getElementById('book-list');
        // Create tr element
        const row = document.createElement('tr');
        // Insert cols
        row.innerHTML = `
        <td>${book.title}</td>
        <td>${book.author}</td>
        <td>${book.isbn}</td>
        <td><a href="#" class="delete">X<a></td>`;
        list.appendChild(row);
    }

    showAlert(message, className) {
        // Create div
        const div = document.createElement('div');
        // Add classes
        div.className = `alert ${className}`;
        // Add text
        div.appendChild(document.createTextNode(message));
        // Get parent
        const container = document.querySelector('.container');
        // Get form
        const form = document.querySelector('#book-form');
        // insert alert
        container.insertBefore(div, form);

        // Timeout after 3 seconds
        setTimeout(function() {
            document.querySelector('.alert').remove();
        }, 3000);
    }

    deleteBook(target) {
        // Remove the book
        if(target.classList.contains('delete')) {
            target.parentElement.parentElement.remove();
        }
    }

    clearFields() {
        // Remove the content of title, author and isbn
        document.getElementById('title').value = '';
        document.getElementById('author').value = '';
        document.getElementById('isbn').value = '';
    }

}

// Local Storage Class
class Storage {
    static getBooks() {
        let books;
        if (localStorage.getItem('books') === null) {
            books = [];
        } else {
            books = JSON.parse(localStorage.getItem('books'));
        }

        return books;
    }

    static displayBooks() {
        const books = Storage.getBooks();

        books.forEach(function(book) {
            const ui = new UI();

            // Add book to UI
            ui.addBookToList(book);
        });
    }

    static addBook(book) {
        const books = Storage.getBooks();

        books.push(book);

        localStorage.setItem('books', JSON.stringify(books));
    }

    static removeBook(isbn) {
        const books = Storage.getBooks();

        books.forEach(function(book, index) {
            if (book.isbn === isbn) {
                books.splice(index, 1);
            }
        });
        
        localStorage.setItem('books', JSON.stringify(books));
    }
}

// Dom Load Event
document.addEventListener('DOMContentLoaded', Storage.displayBooks());

// Event Listeners for add book
document.getElementById('book-form').addEventListener('submit', function(e) {
    // Get forms values
    const title = document.getElementById('title').value,
        author = document.getElementById('author').value,
        isbn = document.getElementById('isbn').value;

    const book = new Book(title, author, isbn);
    const ui = new UI();

    if (title === '' || author === '' || isbn === '') {
        ui.showAlert("Please input the missing fields", 'error');
    } else {
        ui.addBookToList(book);
        // Add to local storage
        Storage.addBook(book);

        ui.showAlert("Book added", 'success');
        ui.clearFields();
    }

    e.preventDefault();
})

// Event Listeners for delete
document.getElementById('book-list').addEventListener('click', 
function(e) {
    const ui = new UI();

    ui.deleteBook(e.target);

    // Remove from LS
    Storage.removeBook(e.target.parentElement.previousElementSibling.textContent);

    // Show message
    ui.showAlert('Book removed', 'success');
    e.preventDefault();
})