import React from 'react'

function BaseCard(props) {

  return (
    <div className="p-4 bg-base-200 rounded-xl drop-shadow hover:drop-shadow-lg outline outline-1 outline-base-content">
        {props.children}
    </div>
  )

}

BaseCard.defaultProps = {
    aspect: 'aspect-square',
}

export default BaseCard