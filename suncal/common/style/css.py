''' Nice Style Sheet for generating HTML reports '''

css = '''
body {
  font-family: sans-serif;
  font-size: 16px;
  line-height: 1.7;
  padding: 1em;
  margin: auto;
  max-width: 56em;
}

a {
  color: #0645ad;
  text-decoration: none;
}

a:visited {
  color: #0b0080;
}

a:hover {
  color: #06e;
}

a:active {
  color: #7c4ca8;
}

a:focus {
  outline: thin dotted;
}

*::selection {
  background: rgba(0, 89, 255, 0.3);
  color: #000;
}

p {
  font-family: sans-serif;
  margin: 1em 0;
}

img {
  max-width: 100%;
}

h1, h2, h3, h4, h5, h6 {
  font-family: sans-serif;
  line-height: 125%;
  margin-top: 2em;
  font-weight: normal;
}

h4, h5, h6 {
  font-weight: bold;
}

h1 {
  font-size: 2.5em;
}

h2 {
  font-size: 2em;
}

h3 {
  font-size: 1.5em;
}

h4 {
  font-size: 1.2em;
}

h5 {
  font-size: 1em;
}

h6 {
  font-size: 0.9em;
}

table {
    margin: 10px 5px;
    border-collapse: collapse;
 }

th {
    background-color: #eee;
    font-weight: bold;
}

th, td {
    border: 1px solid lightgray;
    padding: .2em 1em;
}'''


css_dark = '''
th {
    background-color: #444;
}

th, td {
    border: 1px solid darkgray;
}
'''
