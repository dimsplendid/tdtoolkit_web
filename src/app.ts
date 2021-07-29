import path from 'path'
import express from 'express'
​import {PythonShell, Options} from 'python-shell'

const app = express();
const port = 3000;

// middleware static
// make the root alias (../public -> /)
app.use(express.static(path.join(__dirname, '..', 'public')))

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

app.get('/submit/VHR', (req, res) => {
    res.send(req.query)
})
app.get('/query/total_table', (req, res) => {
    console.log(req.query)
    let options = {
        mode: "text",
        scriptPath: './python_scripts',
        args: [
            req.query.LC,
            req.query.cell_gap_lower,
            req.query.cell_gap_upper,
        ]
    } as Options
    PythonShell.run('query_total_table.py', options, (err, output) => {
        if (err) {
            res.send(err)
        } else {
            res.send(output)
        }
    })
})

app.listen(port, () => {
    if (port === 3000) {
        console.log('true')
    }
    console.log(`server is listening on ${port} !!!`);
});