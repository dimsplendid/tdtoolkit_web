import path from 'path'
import extrac from 'extract-zip'
import { Options } from 'extract-zip'
import { Request, Response, NextFunction, request } from 'express'
import fs from 'fs'
import { fstat } from 'fs';
import formidable from 'formidable'
import { constants } from 'buffer'

const uploadDir = path.join(__dirname, '../public/uploads/');
const extractDir = path.join(__dirname, '../public/extract_files/');

// extractZip
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
        fileName?: string
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
            next(err)
        }
    
        res.send(req.fileName)

        // extract to target directory
    })
    form.on('fileBegin', (name, file) => {
        if (file.name !== null && file.name !== "") {
            // Generate a time stamp zip to prevent duplicate upload files
            const fileName = path.basename(file.name, path.extname(file.name))
            const fileExt = path.extname(file.name)
            req.fileName = `${fileName}_${new Date().getTime()}${fileExt}`
            file.path = path.join(uploadDir, req.fileName)
        }
    })

    form.on('file', (name, file) => {
        if (file.name === null) {
            return res.status(400).json({
                status: "Fail",
                message: "Something wrong",
            })
        } else if (file.name === "") {
            // delete the 'empty' file
            fs.unlink(file.path, (err) => {
                if (err) {
                    console.log(err)
                    throw err
                }
            })
            return res.status(400).json({
                status: "Fail",
                message: "No file",
            })
        }
    })
    // next()
}

function test_print(req: Request, res: Response, next: NextFunction) {

    
}

export { uploadZip }