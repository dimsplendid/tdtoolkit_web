import path from 'path'
import extrac from 'extract-zip'
import {Options} from 'extract-zip'
import {Request, Response, NextFunction} from 'express'
import fs from 'fs'
import {fstat} from 'fs';
import {IncomingForm} from 'formidable'

const uploadDir = path.join(__dirname, '/uploads/');
const extractDir = path.join(__dirname, '/extract_files/');

function extractZip (file:string, destination: string, deleteSource: boolean) {
    extrac(file, {dir: destination})
        .then(state => {
            if (deleteSource) fs.unlinkSync(file)
        })
        .catch(err => console.log(err))
}

function uploadZip (req: Request, res:Response, next: NextFunction) {
    if (!fs.existsSync(uploadDir)) {
        fs.mkdirSync(uploadDir);
    }
    if (!fs.existsSync(extractDir)) {
        fs.mkdirSync(extractDir);
    }
    const form = new IncomingForm()
    form.uploadDir
    form.on('file', (field, file) => {
        if (typeof(file.name) === 'string') {
            file.path = path.join(uploadDir, file.name)
        }
    })

    form.on('end', () => {
        res.send("upload success")
    })
    form.parse(req, (err, fields, files) => {
        if (err) {
            return res.status(500).json({error: err})
        }
        if (Object.keys(files).length === 0) return res.status(400).json({ message: "no files uploaded" });

    })
}

export {uploadZip}