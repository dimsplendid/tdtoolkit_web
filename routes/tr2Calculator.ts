import { Router } from 'express'
import { extractZip, uploadZip } from './lib/zipUtils'
import { update_db, calculate_summary } from './lib/tr2Utils'

const router = Router()

router.post('/upload', [
    uploadZip, 
    extractZip,
    update_db,
    calculate_summary
])

export { router as tr2Calculator }