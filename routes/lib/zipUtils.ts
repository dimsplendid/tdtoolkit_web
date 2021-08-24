import path from 'path'
import extrac from 'extract-zip'
import { Request, Response, NextFunction, request } from 'express'
import fs from 'fs'
import formidable from 'formidable'

// setting upload & extract directory
const uploadDir = path.join(__dirname, '../../public/uploads/');
const extractDir = path.join(__dirname, '../../public/extract_files/');

// adding needed attributes to Request
declare module "express-serve-static-core" {
    interface Request {
        filePath?: string
    }
}

/**
 * Pass the upload zip file with proper name to next middle ware
 * file path is save to req.filePath
 * @param req 
 * @param res 
 * @param next  
 */
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
        next()

        // extract to target directory
    })
    form.on('fileBegin', (name, file) => {
        if (file.name !== null && file.name !== "") {
            // Generate a time stamp zip to prevent duplicate upload files
            const fileName = path.basename(file.name, path.extname(file.name))
            const fileExt = path.extname(file.name)
            const upload_name = `${fileName}_${new Date().getTime()}${fileExt}`
            req.filePath = path.join(uploadDir, upload_name)
            file.path = req.filePath
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
}

/**
 * Extract the zip file at req.filePath
 * and pass the exatract directory path at req.filePath
 * to next middleware.
 * @param req 
 * @param res 
 * @param next 
 */
function extractZip(req: Request, res: Response, next: NextFunction) {
    if (req.filePath === undefined) {
        return res.status(400).json({
            status: "Fail",
            message: "No file path in extractZip",
        })
    }
    const dest = path.join(extractDir, path.basename(req.filePath, path.extname(req.filePath)))
    extrac(req.filePath, { dir: dest })
        .then(() => {
            req.filePath = dest
            next()
        })
        .catch(err => console.log(err))
}

export { uploadZip, extractZip }