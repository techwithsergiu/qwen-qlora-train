import { defineConfig } from 'vitepress'
import { withMermaid } from 'vitepress-plugin-mermaid'

export default withMermaid(
  defineConfig({
    title: 'Qwen QLoRA train',
    description: 'Agent-style SFT / QLoRA pipeline for Qwen3 and Qwen3.5 on consumer hardware.',
    base: '/qwen-qlora-train/',

    head: [
      ['link', { rel: 'icon', href: '/qwen-qlora-train/favicon.ico' }],
    ],

    themeConfig: {
      nav: [
        { text: 'Quickstart', link: '/quickstart' },
        {
          text: 'Training',
          items: [
            { text: 'Training pipeline', link: '/training-pipeline' },
            { text: 'Dataset pipeline', link: '/dataset-pipeline' },
            { text: 'Reasoning control', link: '/reasoning' },
          ],
        },
        {
          text: 'After training',
          items: [
            { text: 'Inference', link: '/inference' },
            { text: 'CPU merge', link: '/merge' },
            { text: 'Post-merge workflow', link: '/post-merge-workflow' },
          ],
        },
        {
          text: 'GitHub',
          link: 'https://github.com/techwithsergiu/qwen-qlora-train',
        },
      ],

      sidebar: [
        {
          text: 'Getting started',
          items: [
            { text: 'Overview', link: '/' },
            { text: 'Setup', link: '/setup' },
            { text: 'Quickstart', link: '/quickstart' },
            { text: 'Commands', link: '/commands' },
          ],
        },
        {
          text: 'Training',
          items: [
            { text: 'Training pipeline', link: '/training-pipeline' },
            { text: 'Dataset pipeline', link: '/dataset-pipeline' },
            { text: 'Reasoning control', link: '/reasoning' },
            { text: 'Config reference', link: '/config-reference' },
          ],
        },
        {
          text: 'After training',
          items: [
            { text: 'Inference', link: '/inference' },
            { text: 'CPU merge', link: '/merge' },
            { text: 'Post-merge workflow', link: '/post-merge-workflow' },
          ],
        },
        {
          text: 'Misc',
          items: [
            { text: 'Troubleshooting', link: '/troubleshooting' },
            { text: 'Third-party Licenses', link: '/THIRD_PARTY_LICENSES' },
          ],
        },
      ],

      socialLinks: [
        {
          icon: 'github',
          link: 'https://github.com/techwithsergiu/qwen-qlora-train',
        },
      ],

      footer: {
        message: 'Released under the Apache 2.0 License.',
      },

      editLink: {
        pattern:
          'https://github.com/techwithsergiu/qwen-qlora-train/edit/main/:path',
        text: 'Edit this page on GitHub',
      },
    },

    mermaid: {
      theme: 'default',
    },
  })
)
