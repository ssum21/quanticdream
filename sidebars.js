/**
 * @type {import('@docusaurus/plugin-content-docs').SidebarsConfig}
 */
const sidebars = {
  tutorialSidebar: [
    {
      type: 'doc',
      id: 'introduction', // docs/00-introduction.md → introduction
      label: '📘 위키 소개',
    },
    {
      type: 'category',
      label: '📚 학습 자료 (Learning)',
      collapsed: false,
      items: [
        {
          type: 'category',
          label: '🧠 딥러닝 스터디',
          items: ['placeholder'], // 준비 중
        },
        {
          type: 'category',
          label: '📐 수학 스터디',
          items: ['placeholder'],
        },
        {
          type: 'category',
          label: '🤖 머신러닝 스터디',
          items: ['learning/ml-study/regression_01'],
        },
      ],
    },
    {
      type: 'category',
      label: '🏆 대회 정리 (Competitions)',
      collapsed: false,
      items: [
        {
          type: 'doc',
          id: 'competitions/KatFishNet Based Hybrid', // 경로 수정됨
          label: 'KatFishNet Hybrid',
        },
        {
          type: 'doc',
          id: 'placeholder',
          label: '2025 SWedu 대회 정리 (준비 중)',
        },
      ],
    },
    {
      type: 'category',
      label: '📊 AlphaQuant Alpha',
      items: ['placeholder'],
    },
    {
      type: 'category',
      label: '👥 팀 및 운영',
      items: ['placeholder'],
    },
  ],
};

module.exports = sidebars;
