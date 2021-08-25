import { Router } from "express"
import { Request, Response, NextFunction } from 'express'
import { extractZip, uploadZip } from "./lib/zipUtils"
import { PythonShell, Options } from "python-shell"

const router = Router()

const axo_exporter = (req: Request, res: Response, next: NextFunction) => {
    // console.log(req.query)
    let options = {
        mode: "text",
        scriptPath: './python_scripts',
        args: [
            req.filePath,
            0.3
        ]
    } as Options
    console.log(req.filePath)

    PythonShell.run('axo_exporter.py', options, (err, output) => {
        if (err) {
            next(err)
        } else if (output) {
            let file_name = output[output.length-1]
            res.download(file_name)
        } else {
            return res.status(400).json({
                status: "Fail",
                message: "something wrong at axo_exporter.py, didn't get file",
            })
        }
    })
}

router.post('/axo_exporter',[
    uploadZip,
    extractZip,
    axo_exporter
])

export { router as axoExporter }