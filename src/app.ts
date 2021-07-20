import express from 'express';
​
const app = express();
const port = 3000;

app.get('/style.css', function(req, res) {
    res.sendFile("public/css/style.css", {root: '.'});
});

app.get('/', (req, res) => {
    // res.send('The server is working!');
    res.sendFile('public/index.html', {root: '.'})
});

// for insert vender measuring data
// Something weird here, need figure out later
// the action in "/submit/form" would just in "/submit/*" ? 
app.get('/submit/form', (req, res) => {
    res.sendFile('public/checklist_sheet.html', {root: '.'})
})
app.get('/submit/VHR', (req, res) => {
    res.send(req.query)
})


app.listen(port, () => {
    if (port === 3000) {
        console.log('true')
    }
    console.log(`server is listening on ${port} !!!`);
});