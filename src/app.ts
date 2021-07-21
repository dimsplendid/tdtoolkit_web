import express from 'express'
​import {PythonShell, Options} from 'python-shell'

const app = express();
const port = 3000;

app.get('/style.css', function(req, res) {
    res.sendFile("public/css/style.css", {root: '.'});
});

app.get('/', (req, res) => {
    // res.send('The server is working!');
    res.sendFile('public/index.html', {root: '.'})
});

app.get('/plot/VT', (req, res) => {
    let options = {
        mode: "text",
        scriptPath: './python_scripts',
        args: [
            req.query.LC,
            req.query.cell_gap,
            req.query.V_max,
            req.query.V_min
        ]
    } as Options
    PythonShell.run('draw_VT.py', options, (err, output) => {
        if (err) {
            res.send(err)
        } else {
            res.send(output)
        }
    })
})

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