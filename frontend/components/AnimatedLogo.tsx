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
        src="/nice-fade.png"
        alt="KARGA"
        className="w-auto h-auto max-w-[95%] md:max-w-6xl lg:max-w-7xl object-contain -mb-8 md:-mb-20"
        style={{ maxHeight: '405px', transform: 'translateX(-8%)' }}
      />
      <style jsx>{`
        @media (min-width: 768px) {
          img {
            max-height: 840px !important;
          }
        }
      `}</style>
    </motion.div>
  )
}
