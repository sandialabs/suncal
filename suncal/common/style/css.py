''' Nice Style Sheet for generating HTML reports '''

css = '''
body {
  color: #444;
  font-family: Georgia, Palatino, 'Palatino Linotype', Times, 'Times New Roman', serif;
  font-size: 16px;
  line-height: 1.7;
  padding: 1em;
  margin: auto;
  max-width: 56em;
  background: #fefefe;
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
  margin: 1em 0;
}

img {
  max-width: 100%;
}

h1, h2, h3, h4, h5, h6 {
  color: #111;
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
  margin-bottom: 2em;
  border-bottom: 1px solid #ddd;
  border-right: 1px solid #ddd;
  border-spacing: 0;
  border-collapse: collapse;
}

table th {
  padding: .2em 1em;
  background-color: #eee;
  border-top: 1px solid #ddd;
  border-left: 1px solid #ddd;
}

table td {
  padding: .2em 1em;
  border-top: 1px solid #ddd;
  border-left: 1px solid #ddd;
  vertical-align: top;
}'''
