
import { Request, Response, NextFunction } from 'express'
import { PythonShell, Options } from 'python-shell'

// adding needed attributes to Request
declare module "express-serve-static-core" {
    interface Request {
        batch?: string
    }
}

// updata_db
const update_db = (req: Request, res: Response, next: NextFunction) => {
    let options = {
        mode: "text",
        scriptPath: './python_scripts',
        args: [
            req.filePath
        ]
    } as Options
    PythonShell.run('update_db.py', options, (err, output) => {
        if (err) {
            return res.status(400).json({
                status: "Fail",
                message: err,
            })
        } else if (output) {
            req.batch = output[output?.length - 1]
            next()
        } else {
            return res.status(400).json({
                status: "Fail",
                message: "something wrong when updata_db.py, didn't generate batch number",
            })
        }
    })
}

// calculate_summary
const calculate_summary = (req: Request, res: Response, next: NextFunction) => {
    let options = {
        mode: "text",
        scriptPath: './python_scripts',
        args: [
            req.batch
        ]
    } as Options
    PythonShell.run('calculate_summary.py', options, (err, output) => {
        if (err) {
            res.send(err)
        } else if (output) {
            let file_name = output[output?.length - 1] // need modified later
            res.download(file_name)
        } else {
            return res.status(400).json({
                status: "Fail",
                message: "something wrong when updata_db.py, didn't generate batch number",
            })
        }
    })
}

export { update_db, calculate_summary }