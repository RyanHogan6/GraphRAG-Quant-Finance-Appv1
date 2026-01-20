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
        src="/updated-logo-2.1.png"
        alt="KARGA"
        className="w-auto h-auto max-w-[90%] md:max-w-4xl lg:max-w-5xl object-contain"
        style={{ maxHeight: '270px', transform: 'translateX(-8%)' }}
      />
      <style jsx>{`
        @media (min-width: 768px) {
          img {
            max-height: 560px !important;
          }
        }
      `}</style>
    </motion.div>
  )
}
