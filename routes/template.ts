import express from 'express'
import {Request, Response, NextFunction} from 'express'
import {PythonShell, Options} from 'python-shell'

const router = express.Router()

const template = (req: Request, res: Response, next: NextFunction) => {
    res.send('template router')
}

const pythonTemplate = (req: Request, res: Response, next: NextFunction) => {
    console.log(req.query)
    let options = {
        mode: "text",
        scriptPath: './python_scripts',
        args: [
            // keywords, can come from req.query.[keywords]
            "template python"
        ]
    } as Options
    PythonShell.run('template.py', options, (err, output) => {
        if (err) {
            res.send(err)
        } else {
            res.send(output)
        }
    })
}

router.get('/template', template)
router.get('/templatePython', pythonTemplate)

export {router as templateRouter}