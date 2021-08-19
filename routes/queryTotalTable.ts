import express from 'express'
import {Request, Response, NextFunction} from 'express'
import {PythonShell, Options} from 'python-shell'

const router = express.Router()

// middleware that is specific to this router

router.get('/query/total_table', (req, res) => {
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
            // need learn how to deal with undefined/null later
            let file_name = output![output!.length - 1]
            res.download(file_name)
        }
    })
})

export {router as queryTotalTable}