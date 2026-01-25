'use client'

import { motion } from 'framer-motion'

export default function AnimatedLogo() {
  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.8, delay: 0.2 }}
      className="flex items-center justify-center"
    >
      <img
        src="/updated-logo2.2.png"
        alt="KARGA"
        className="w-auto h-auto object-contain"
        style={{ maxHeight: '70px' }}
      />
    </motion.div>
  )
}
