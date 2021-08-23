import path from 'path'
import extrac from 'extract-zip'
import { Options } from 'extract-zip'
import { Request, Response, NextFunction, request } from 'express'
import fs from 'fs'
import { fstat } from 'fs';
import formidable from 'formidable'

const uploadDir = path.join(__dirname, '../public/uploads/');
const extractDir = path.join(__dirname, '../public/extract_files/');

function extractZip(file: string, destination: string, deleteSource: boolean) {
    extrac(file, { dir: destination })
        .then(state => {
            if (deleteSource) fs.unlinkSync(file)
        })
        .catch(err => console.log(err))
}

// adding needed attributes to Request
declare module "express-serve-static-core" {
    interface Request {
        fileName?: String
    }
}

// upload zip file, and pass the un-archived folder to next middleware
function uploadZip(req: Request, res: Response, next: NextFunction) {
    if (!fs.existsSync(uploadDir)) {
        fs.mkdirSync(uploadDir)
    }
    if (!fs.existsSync(extractDir)) {
        fs.mkdirSync(extractDir)
    }

    const form = formidable({
        multiples: false, // Only one file each time
        uploadDir: uploadDir,
        maxFileSize: 100 * 1024 * 1024, // file size limit 100 MB
        keepExtensions: true
    })

    form.parse(req, (err, field, files) => {
        console.log(field)
        console.log(files)
        if (err) {
            console.log("Error parsing the files");
            return res.status(400).json({
                status: "Fail",
                message: "There was an error parsing the files",
                error: err,
            })
        }
        
        // file upload detect
        if (Object.keys(files).length === 0) {
            return res.status(400).json({
                status: "Fail",
                message: "no files uploaded",
                error: err,
            })
        }

        // res.send(files.name) // this would never end, interesting
        
        res.send(req.fileName)

        // // no file uploads
        // if (files.name === undefined) {
        //     return res.status(400).json({
        //         status: "Fail",
        //         message: "No file uploaded",
        //         error: err
        //     })
        // }
        // // check whether uploaded file is zip
        // else if (!(path.extname((files.name).toString()) === 'zip')) {
        //     return res.status(400).json({
        //         status: "Fail",
        //         message: "Unsupported file type",
        //         error: err
        //     })
        // }
    })
    // form.on('end', () => {
    //     res.status(200).json({
    //         status: "Success",
    //         message: "upload success"
    //     })
    // })

    form.on('file', (name, file) => {
        if (file.name === null) {
            return res.status(400).json({
                status: "Fail",
                message: "Something wrong",
            })
        } else if (file.name === "") {
            return res.status(400).json({
                status: "Fail",
                message: "No file",
            })
        }
        req.fileName = path.basename(file.name, path.extname(file.name))
    })
    // next()
}

export { uploadZip }